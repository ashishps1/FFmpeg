/*
 * Copyright (c) 2017 Ronald S. Bultje <rsbultje@gmail.com>
 * Copyright (c) 2017 Ashish Pratap Singh <ashk43712@gmail.com>
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
 * Calculate Motion score between two input videos.
 */

#include <inttypes.h>
#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "dualinput.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "motion.h"
#include "video.h"
#include "convolution.h"

typedef struct MOTIONContext {
    const AVClass *class;
    FFDualInputContext dinput;
    int width;
    int height;
    uint8_t type;
    float *ref_data;
    float *prev_blur_data;
    float *blur_data;
    float *temp_data;
    double motion_sum;
    uint64_t nb_frames;
} MOTIONContext;

#define OFFSET(x) offsetof(MOTIONContext, x)
#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define OPT_RANGE_PIXEL_OFFSET (-128)

static const AVOption motion_options[] = {
    { NULL }
};

AVFILTER_DEFINE_CLASS(motion);

static const float FILTER_5[5] = {
    0.054488685,
    0.244201342,
    0.402619947,
    0.244201342,
    0.054488685
};

static inline double get_motion_avg(double motion_sum, uint64_t nb_frames)
{
    return motion_sum / nb_frames;
}

static void offset(MOTIONContext *s, const AVFrame *ref, int stride)
{
    int w = s->width;
    int h = s->height;
    int i,j;

    int ref_stride = ref->linesize[0];

    uint8_t *ref_ptr = ref->data[0];

    float *ref_ptr_data = s->ref_data;

    for(i = 0; i < h; i++) {
        for(j = 0; j < w; j++) {
            ref_ptr_data[j] = (float) ref_ptr[j] + OPT_RANGE_PIXEL_OFFSET;
        }
        ref_ptr += ref_stride / sizeof(uint8_t);
        ref_ptr_data += stride / sizeof(float);
    }
}

static double image_sad_c(const float *img1, const float *img2, int w,
                          int h, int img1_stride, int img2_stride)
{
    float accum = (float)0.0;

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float img1px = img1[i * img1_stride + j];
            float img2px = img2[i * img2_stride + j];

            accum += fabs(img1px - img2px);
        }
    }

    return (float) (accum / (w * h));
}

static int compute_motion(const float *ref, const float *dis, int w, int h,
                          int ref_stride, int dis_stride, double *score,
                          void *ctx)
{
    *score = image_sad_c(ref, dis, w, h, ref_stride / sizeof(float),
                         dis_stride / sizeof(float));

    return 0;
}

static void set_meta(AVDictionary **metadata, const char *key, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%0.2f", d);
    av_dict_set(metadata, key, value, 0);
}

static AVFrame *do_motion(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    MOTIONContext *s = ctx->priv;
    AVDictionary **metadata = &main->metadata;
    int stride;
    size_t data_sz;
    double score;

    stride = ALIGN_CEIL(s->width * sizeof(float));
    data_sz = (size_t)stride * s->height;

    offset(s, ref, stride);

    convolution_f32_c(FILTER_5, 5, s->ref_data, s->blur_data, s->temp_data,
                      s->width, s->height, stride / sizeof(float), stride /
                      sizeof(float));

    if(!s->nb_frames) {
        score = 0.0;
    } else {
        compute_motion(s->prev_blur_data, s->blur_data, s->width, s->height,
                       stride, stride, &score, s);
    }

    memcpy(s->prev_blur_data, s->blur_data, data_sz);

    set_meta(metadata, "lavfi.motion.score", score);

    s->nb_frames++;

    s->motion_sum += score;

    return main;
}

static av_cold int init(AVFilterContext *ctx)
{
    MOTIONContext *s = ctx->priv;

    s->dinput.process = do_motion;

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
    MOTIONContext *s = ctx->priv;
    int stride;
    size_t data_sz;

    if (ctx->inputs[0]->w != ctx->inputs[1]->w ||
        ctx->inputs[0]->h != ctx->inputs[1]->h) {
        av_log(ctx, AV_LOG_ERROR, "Width and height of input videos must be same.\n");
        return AVERROR(EINVAL);
    }
    if (ctx->inputs[0]->format != ctx->inputs[1]->format) {
        av_log(ctx, AV_LOG_ERROR, "Inputs must be of same pixel format.\n");
        return AVERROR(EINVAL);
    }

    s->width = ctx->inputs[0]->w;
    s->height = ctx->inputs[0]->h;

    stride = ALIGN_CEIL(s->width * sizeof(float));
    data_sz = (size_t)stride * s->height;

    if (!(s->ref_data = av_malloc(data_sz))) {
        av_log(ctx, AV_LOG_ERROR, "ref_buf allocation failed.\n");
        return AVERROR(EINVAL);
    }
    if (!(s->prev_blur_data = av_mallocz(data_sz))) {
        av_log(ctx, AV_LOG_ERROR, "prev_blur_buf allocation failed.\n");
        return AVERROR(EINVAL);
    }
    if (!(s->blur_data = av_mallocz(data_sz))) {
        av_log(ctx, AV_LOG_ERROR, "blur_buf allocation failed.\n");
        return AVERROR(EINVAL);
    }
    if (!(s->temp_data = av_mallocz(data_sz * 2))) {
        av_log(ctx, AV_LOG_ERROR, "temp_buf allocation failed.\n");
        return AVERROR(EINVAL);
    }

    s->type = desc->comp[0].depth > 8 ? 10 : 8;

    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    MOTIONContext *s = ctx->priv;
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
    MOTIONContext *s = inlink->dst->priv;
    return ff_dualinput_filter_frame(&s->dinput, inlink, inpicref);
}

static int request_frame(AVFilterLink *outlink)
{
    MOTIONContext *s = outlink->src->priv;
    return ff_dualinput_request_frame(&s->dinput, outlink);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    MOTIONContext *s = ctx->priv;

    ff_dualinput_uninit(&s->dinput);

    av_free(s->ref_data);
    av_free(s->prev_blur_data);
    av_free(s->blur_data);
    av_free(s->temp_data);

    av_log(ctx, AV_LOG_INFO, "MOTION AVG: %.3f\n", get_motion_avg(s->motion_sum,
                                                                  s->nb_frames));
}

static const AVFilterPad motion_inputs[] = {
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

static const AVFilterPad motion_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
        .request_frame = request_frame,
    },
    { NULL }
};

AVFilter ff_vf_motion = {
    .name          = "motion",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the MOTION between two video streams."),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(MOTIONContext),
    .priv_class    = &motion_class,
    .inputs        = motion_inputs,
    .outputs       = motion_outputs,
};
