/*
 * Copyright (c) 2011 Roger Pau Monné <roger.pau@entel.upc.edu>
 * Copyright (c) 2011 Stefano Sabatini
 * Copyright (c) 2013 Paul B Mahol
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Caculate the VMAF between two input videos.
 */

#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "dualinput.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "vmaf.h"
#include "video.h"

typedef struct VMAFContext {
    const AVClass *class;
    FFDualInputContext dinput;
    double mse, min_mse, max_mse, mse_comp[4];
    uint64_t nb_frames;
    FILE *stats_file;
    char *stats_file_str;
    int stats_version;
    int stats_header_written;
    int stats_add_max;
    int max[4], average_max;
    int is_rgb;
    uint8_t rgba_map[4];
    char comps[4];
    int nb_components;
    int planewidth[4];
    int planeheight[4];
    double planeweight[4];
    VMAFDSPContext dsp;
} VMAFContext;

#define OFFSET(x) offsetof(VMAFContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption vmaf_options[] = {
    {"stats_file", "Set file where to store per-frame difference information", OFFSET(stats_file_str), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS },
    {"f",          "Set file where to store per-frame difference information", OFFSET(stats_file_str), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS },
    {"stats_version", "Set the format version for the stats file.",               OFFSET(stats_version),  AV_OPT_TYPE_INT,    {.i64=1},    1, 2, FLAGS },
    {"output_max",  "Add raw stats (max values) to the output log.",            OFFSET(stats_add_max), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    { NULL }
};

AVFILTER_DEFINE_CLASS(vmaf);

static inline unsigned pow_2(unsigned base)
{
    return base*base;
}

static inline double get_vmaf(double mse, uint64_t nb_frames, int max)
{
    return 10.0 * log10(pow_2(max) / (mse / nb_frames));
}

static uint64_t sse_line_8bit(const uint8_t *main_line,  const uint8_t *ref_line, int outw)
{
    int j;
    unsigned m2 = 0;

    for (j = 0; j < outw; j++)
        m2 += pow_2(main_line[j] - ref_line[j]);

    return m2;
}

static uint64_t sse_line_16bit(const uint8_t *_main_line, const uint8_t *_ref_line, int outw)
{
    int j;
    uint64_t m2 = 0;
    const uint16_t *main_line = (const uint16_t *) _main_line;
    const uint16_t *ref_line = (const uint16_t *) _ref_line;

    for (j = 0; j < outw; j++)
        m2 += pow_2(main_line[j] - ref_line[j]);

    return m2;
}

static inline
void compute_images_mse(VMAFContext *s,
                        const uint8_t *main_data[4], const int main_linesizes[4],
                        const uint8_t *ref_data[4], const int ref_linesizes[4],
                        int w, int h, double mse[4])
{
    int i, c;
}

static void set_meta(AVDictionary **metadata, const char *key, char comp, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%0.2f", d);
    if (comp) {
        char key2[128];
        snprintf(key2, sizeof(key2), "%s%c", key, comp);
        av_dict_set(metadata, key2, value, 0);
    } else {
        av_dict_set(metadata, key, value, 0);
    }
}

static AVFrame *do_vmaf(AVFilterContext *ctx, AVFrame *main,
                        const AVFrame *ref)
{
    VMAFContext *s = ctx->priv;
    double comp_mse[4], mse = 0;
    int j, c;
    AVDictionary **metadata = &main->metadata;

    compute_images_mse(s, (const uint8_t **)main->data, main->linesize,
                          (const uint8_t **)ref->data, ref->linesize,
                          main->width, main->height, comp_mse);

    return main;
}

static av_cold int init(AVFilterContext *ctx)
{
    VMAFContext *s = ctx->priv;

    s->min_mse = +INFINITY;
    s->max_mse = -INFINITY;

    if (s->stats_file_str) {
        if (s->stats_version < 2 && s->stats_add_max) {
            av_log(ctx, AV_LOG_ERROR,
                "stats_add_max was specified but stats_version < 2.\n" );
            return AVERROR(EINVAL);
        }
        if (!strcmp(s->stats_file_str, "-")) {
            s->stats_file = stdout;
        } else {
            s->stats_file = fopen(s->stats_file_str, "w");
            if (!s->stats_file) {
                int err = AVERROR(errno);
                char buf[128];
                av_strerror(err, buf, sizeof(buf));
                av_log(ctx, AV_LOG_ERROR, "Could not open stats file %s: %s\n",
                       s->stats_file_str, buf);
                return err;
            }
        }
    }

    s->dinput.process = do_vmaf;
    return 0;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV420P10LE,
        AV_PIX_FMT_NONE
    };

    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

static int config_input_ref(AVFilterLink *inlink)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    AVFilterContext *ctx  = inlink->dst;
    VMAFContext *s = ctx->priv;
    double average_max;
    unsigned sum;
    int j;

    s->nb_components = desc->nb_components;
    if (ctx->inputs[0]->w != ctx->inputs[1]->w ||
        ctx->inputs[0]->h != ctx->inputs[1]->h) {
        av_log(ctx, AV_LOG_ERROR, "Width and height of input videos must be same.\n");
        return AVERROR(EINVAL);
    }
    if (ctx->inputs[0]->format != ctx->inputs[1]->format) {
        av_log(ctx, AV_LOG_ERROR, "Inputs must be of same pixel format.\n");
        return AVERROR(EINVAL);
    }


    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    VMAFContext *s = ctx->priv;
    AVFilterLink *mainlink = ctx->inputs[0];
    int ret;

    outlink->w = mainlink->w;
    outlink->h = mainlink->h;
    outlink->time_base = mainlink->time_base;
    outlink->sample_aspect_ratio = mainlink->sample_aspect_ratio;
    outlink->frame_rate = mainlink->frame_rate;
    if ((ret = ff_dualinput_init(ctx, &s->dinput)) < 0)
        return ret;

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *inpicref)
{
    VMAFContext *s = inlink->dst->priv;
    return ff_dualinput_filter_frame(&s->dinput, inlink, inpicref);
}

static int request_frame(AVFilterLink *outlink)
{
    VMAFContext *s = outlink->src->priv;
    return ff_dualinput_request_frame(&s->dinput, outlink);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    VMAFContext *s = ctx->priv;

    ff_dualinput_uninit(&s->dinput);

    if (s->stats_file && s->stats_file != stdout)
        fclose(s->stats_file);
}

static const AVFilterPad vmaf_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },{
        .name         = "reference",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_input_ref,
    },
    { NULL }
};

static const AVFilterPad vmaf_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
        .request_frame = request_frame,
    },
    { NULL }
};

AVFilter ff_vf_vmaf = {
    .name          = "vmaf",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the VMAF between two video streams."),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(VMAFContext),
    .priv_class    = &vmaf_class,
    .inputs        = vmaf_inputs,
    .outputs       = vmaf_outputs,
};
