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
 * Caculate the VMAF between two input videos.
 */

#include <inttypes.h>
#include <pthread.h>
#include <libvmaf.h>
#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "dualinput.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

typedef struct VMAFContext {
    const AVClass *class;
    FFDualInputContext dinput;
    char *format;
    int width;
    int height;
    double curr_vmaf_score;
    double vmaf_score;
    uint64_t nb_frames;
    pthread_t vmaf_thread;
    pthread_attr_t attr;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int eof;
    AVFrame *gmain;
    AVFrame *gref;
    char *model_path;
    char *log_path;
    char *log_fmt;
    int disable_clip;
    int disable_avx;
    int enable_transform;
    int phone_model;
    int psnr;
    int ssim;
    int ms_ssim;
    char *pool;
    FILE *stats_file;
    char *stats_file_str;
    int stats_version;
    int stats_header_written;
    int stats_add_max;
    int nb_components;
} VMAFContext;

#define OFFSET(x) offsetof(VMAFContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption vmaf_options[] = {
    {"stats_file", "Set file where to store per-frame difference information", OFFSET(stats_file_str), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS },
    {"f",          "Set file where to store per-frame difference information", OFFSET(stats_file_str), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS },
    {"stats_version", "Set the format version for the stats file.",               OFFSET(stats_version),  AV_OPT_TYPE_INT,    {.i64=1},    1, 2, FLAGS },
    {"output_max",  "Add raw stats (max values) to the output log.",            OFFSET(stats_add_max), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"model_path",  "Set the model to be used for computing vmaf.",            OFFSET(model_path), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 1, FLAGS},
    {"log_path",  "Set the file path to be used to store logs.",            OFFSET(log_path), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 1, FLAGS},
    {"log_fmt",  "Set the format of the log (xml or json).",            OFFSET(log_fmt), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 1, FLAGS},
    {"disable_clip",  "Disables clip for computing vmaf.",            OFFSET(disable_clip), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"disable avx",  "Disables avx for computing vmaf.",            OFFSET(disable_avx), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"enable_transform",  "Enables transform for computing vmaf.",            OFFSET(enable_transform), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"phone_model",  "Invokes the phone model that will generate higher VMAF scores.",            OFFSET(phone_model), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"psnr",  "Enables computing psnr along with vmaf.",            OFFSET(psnr), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"ssim",  "Enables computing ssim along with vmaf.",            OFFSET(ssim), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"ms_ssim",  "Enables computing ms-ssim along with vmaf.",            OFFSET(ms_ssim), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"pool",  "Set the pool method to be used for computing vmaf.",            OFFSET(pool), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 1, FLAGS},
    { NULL }
};

AVFILTER_DEFINE_CLASS(vmaf);

static int read_frame(float *ref_data, float *main_data, int stride, double *score, void *ctx){
    VMAFContext *s = (VMAFContext *)ctx;

    if (s->eof) {
        return 1;
    }

    pthread_mutex_lock(&s->lock);

    while (s->gref == NULL) {
        pthread_cond_wait(&s->cond, &s->lock);
    }

    int ref_stride = s->gref->linesize[0];
    int main_stride = s->gmain->linesize[0];
    
    uint8_t *ptr = s->gref->data[0];    
    float *ptr1 = ref_data;
    
    int h = s->height;
    int w = s->width;
    
    int i,j;
    
    ptr = s->gref->data[0];
   
    for (i = 0; i < h; i++) {
        for ( j = 0; j < w; j++) {
            ptr1[j] = (float)ptr[j];
        }
        ptr += ref_stride;
        ptr1 += stride/sizeof(*ptr1);
    }

    ptr = s->gmain->data[0];
    ptr1 = main_data;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            ptr1[j] = (float)ptr[j];
        }
        ptr += main_stride;
        ptr1 += stride/sizeof(*ptr1);
    }

    s->gref = NULL;
    s->gmain = NULL;

    pthread_cond_signal(&s->cond);
    pthread_mutex_unlock(&s->lock);

    return s->eof;
}

static void compute_vmaf_score(VMAFContext *s)
{
    s->vmaf_score = compute_vmaf(s->format, s->width, s->height, read_frame,
                                 s->model_path, s->log_path, s->log_fmt, s->disable_clip,
                                 s->disable_avx, s->enable_transform, s->phone_model,
                                 s->psnr, s->ssim, s->ms_ssim, s->pool, s);
}

static void *call_vmaf(void *ctx)
{
    VMAFContext *s = (VMAFContext *)ctx;
    compute_vmaf_score(s);
    pthread_exit(NULL);
}

static AVFrame *do_vmaf(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    VMAFContext *s = ctx->priv;

    pthread_mutex_lock(&s->lock);
    while (s->gref != NULL) {
        pthread_cond_wait(&s->cond, &s->lock);
    }

    s->gref = malloc(sizeof(AVFrame));
    s->gmain = malloc(sizeof(AVFrame));

    memcpy(s->gref, ref, sizeof(AVFrame));
    memcpy(s->gmain, main, sizeof(AVFrame));  

    int ref_stride = s->gref->linesize[0];
    int main_stride = s->gmain->linesize[0];
    
    pthread_cond_signal(&s->cond);
    pthread_mutex_unlock(&s->lock);

    return main;
}

static av_cold int init(AVFilterContext *ctx)
{
    VMAFContext *s = ctx->priv;

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
    if (!(s->model_path)) {
        av_log(ctx, AV_LOG_ERROR, "No model specified.\n");
        return AVERROR(EINVAL);
    }

    s->format = av_get_pix_fmt_name(ctx->inputs[0]->format);
    s->width = ctx->inputs[0]->w;
    s->height = ctx->inputs[0]->h;

    pthread_mutex_init(&s->lock, NULL);
    pthread_cond_init (&s->cond, NULL);

    pthread_attr_init(&s->attr);

    int rc = pthread_create(&s->vmaf_thread, &s->attr, call_vmaf, (void *)s);
    if (rc) {
        av_log(ctx, AV_LOG_ERROR, "Thread creation failed.\n");
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

    static int ptr = 0;
    if (ptr) {
        s->eof = 1;
        pthread_join(s->vmaf_thread, NULL);
        av_log(ctx, AV_LOG_INFO, "VMAF score: %f\n",s->vmaf_score);
    }
    ptr = 1;
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
    .description   = NULL_IF_CONFIG_SMALL("Calculate the VMAF between two videos."),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(VMAFContext),
    .priv_class    = &vmaf_class,
    .inputs        = vmaf_inputs,
    .outputs       = vmaf_outputs,
};
