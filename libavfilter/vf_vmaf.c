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
 * Calculate the VMAF between two input videos.
 */

#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "dualinput.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "video.h"
#include "adm.h"
#include "motion.h"
#include "vif.h"
#include "vmaf.h"

typedef struct VMAFContext {
    const AVClass *class;
    FFDualInputContext dinput;
    const AVPixFmtDescriptor *desc;
    int width;
    int height;
    uint8_t called;
    double score;
    double scores[8];
    double score_num;
    double score_den;
    float *ref_data;
    float *main_data;
    float *adm_data_buf;
    float *adm_temp_lo;
    float *adm_temp_hi;
    float *prev_blur_data;
    float *blur_data;
    float *temp_data;
    float *vif_data_buf;
    float *vif_temp;
    double vmaf_sum;
    double prev_motion_score;
    uint64_t nb_frames;
    char *model_path;
    char *log_path;
    char *log_fmt;
    int enable_transform;
    int phone_model;
    char *pool;    
    DArray adm_array,
           adm_scale_array[4],
           motion_array,
           motion2_array,
           vif_scale_array[4],
           vif_array;
} VMAFContext;

#define OFFSET(x) offsetof(VMAFContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption vmaf_options[] = {
    {"model_path",  "Set the model to be used for computing vmaf.",                     OFFSET(model_path), AV_OPT_TYPE_STRING, {.str="/usr/local/share/model/vmaf_v0.6.1.pkl"}, 0, 1, FLAGS},
    {"log_path",  "Set the file path to be used to store logs.",                        OFFSET(log_path), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 1, FLAGS},
    {"log_fmt",  "Set the format of the log (xml or json).",                            OFFSET(log_fmt), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 1, FLAGS},
    {"enable_transform",  "Enables transform for computing vmaf.",                      OFFSET(enable_transform), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"phone_model",  "Invokes the phone model that will generate higher VMAF scores.",  OFFSET(phone_model), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"pool",  "Set the pool method to be used for computing vmaf.",                     OFFSET(pool), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 1, FLAGS},
    { NULL }
};

AVFILTER_DEFINE_CLASS(vmaf);

#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define OPT_RANGE_PIXEL_OFFSET (-128)
#define INIT_FRAMES 1000
#define ADM2_CONSTANT 0.0
#define ADM_SCALE_CONSTANT 0.0

void init_arr(DArray *a, size_t init_size)
{
    a->array = (double *) av_malloc(init_size * sizeof(double));
    a->used = 0;
    a->size = init_size;
}

void append_array(DArray *a, double e)
{
    if (a->used == a->size) {
        a->size *= 2;
        a->array = (double *) av_realloc(a->array, a->size * sizeof(double));
    }
    a->array[a->used++] = e;
}

double get_at_pos(DArray *a, int pos)
{
    return a->array[pos];
}

void free_arr(DArray *a)
{
    av_free(a->array);
    a->array = NULL;
    a->used = a->size = 0;
}

#define offset_fn(type, bits) \
    static void offset_##bits##bit(VMAFContext *s, const AVFrame *ref, AVFrame *main, int stride) \
{ \
    int w = s->width; \
    int h = s->height; \
    int i,j; \
    \
    int ref_stride = ref->linesize[0]; \
    int main_stride = main->linesize[0]; \
    \
    const type *ref_ptr = (const type *) ref->data[0]; \
    const type *main_ptr = (const type *) main->data[0]; \
    \
    float *ref_ptr_data = s->ref_data; \
    float *main_ptr_data = s->main_data; \
    \
    for(i = 0; i < h; i++) { \
        for(j = 0; j < w; j++) { \
            ref_ptr_data[j] = (float) ref_ptr[j] + OPT_RANGE_PIXEL_OFFSET; \
            main_ptr_data[j] = (float) main_ptr[j] + OPT_RANGE_PIXEL_OFFSET; \
        } \
        ref_ptr += ref_stride / sizeof(type); \
        ref_ptr_data += stride / sizeof(float); \
        main_ptr += main_stride / sizeof(type); \
        main_ptr_data += stride / sizeof(float); \
    } \
}

offset_fn(uint8_t, 8);
offset_fn(uint16_t, 10);

static int compute_vmaf(const float *ref, const float *main, int w, int h,
                        int ref_stride, int main_stride, void *ctx)
{
    VMAFContext *s = (VMAFContext *) ctx;

    int stride;
    size_t data_sz;
    int i,j;
 
    stride = ALIGN_CEIL(s->width * sizeof(float));
    data_sz = (size_t)stride * s->height;

    compute_adm1(s->ref_data, s->main_data, w, h, stride, stride, &s->score,
                 &s->score_num, &s->score_den, s->scores, s->adm_data_buf,
                 s->adm_temp_lo, s->adm_temp_hi);

    append_array(&s->adm_array, ((s->score_num + ADM_SCALE_CONSTANT) / (s->score_den + ADM_SCALE_CONSTANT)));
    j = 0;
    for(i = 0; j < 4; i += 2) {
        append_array(&s->adm_scale_array[j], ((s->scores[i] + ADM_SCALE_CONSTANT) / (s->scores[i+1] + ADM_SCALE_CONSTANT)));
        j++;
    }
    printf("adm : %.3f ",s->score);

    convolution_f32(FILTER_5, 5, s->ref_data, s->blur_data, s->temp_data,
                    s->width, s->height, stride / sizeof(float), stride /
                    sizeof(float));

    if(!s->nb_frames) {
        s->score = 0.0;
    } else {
        compute_motion1(s->prev_blur_data, s->blur_data, s->width, s->height,
                        stride, stride, &s->score);
    }

    memcpy(s->prev_blur_data, s->blur_data, data_sz);

    append_array(&s->motion_array, s->score);
    
    if(s->nb_frames > 0) {
        append_array(&s->motion2_array, FFMIN(s->prev_motion_score, s->score));
    }
    
    s->prev_motion_score = s->score;
    printf("motion : %.3f ",s->score);

    compute_vif1(s->ref_data, s->main_data, w, h, stride, stride, &s->score,
                 &s->score_num, &s->score_den, s->scores, s->vif_data_buf,
                 s->vif_temp);
    j = 0;
    for(i = 0; j < 4; i += 2) {
        append_array(&s->vif_scale_array[j], ((s->scores[i]) / (s->scores[i+1])));
        j++;
    }
    append_array(&s->vif_array, s->score);
    printf("vif : %.3f\n",s->score);

    return 0;
}

static AVFrame *do_vmaf(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    VMAFContext *s = ctx->priv;

    int stride;
    int w = s->width;
    int h = s->height;

    stride = ALIGN_CEIL(w * sizeof(float));

    /** Offset ref and main pixel by OPT_RANGE_PIXEL_OFFSET */
    if (s->desc->comp[0].depth <= 8) {
        offset_8bit(s, ref, main, stride);
    } else {
        offset_10bit(s, ref, main, stride);
    }

    compute_vmaf(s->ref_data, s->main_data, w, h, stride, stride, s);

    s->nb_frames++;

    return main;
}

static av_cold int init(AVFilterContext *ctx)
{
    VMAFContext *s = ctx->priv;

    if(!s->called) {
        int i;
        init_arr(&s->adm_array, INIT_FRAMES);
        for(i = 0; i < 4; i++) {
            init_arr(&s->adm_scale_array[i], INIT_FRAMES);
        }
        init_arr(&s->motion_array, INIT_FRAMES);
        init_arr(&s->motion2_array, INIT_FRAMES);
        for(i = 0; i < 4; i++) {
            init_arr(&s->vif_scale_array[i], INIT_FRAMES);
        }
        init_arr(&s->vif_array, INIT_FRAMES);
    }

    s->called = 1;
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
    AVFilterContext *ctx  = inlink->dst;
    VMAFContext *s = ctx->priv;
    int stride;
    size_t data_sz;
    int adm_buf_stride;
    size_t adm_buf_sz;

    if (ctx->inputs[0]->w != ctx->inputs[1]->w ||
        ctx->inputs[0]->h != ctx->inputs[1]->h) {
        av_log(ctx, AV_LOG_ERROR, "Width and height of input videos must be same.\n");
        return AVERROR(EINVAL);
    }
    if (ctx->inputs[0]->format != ctx->inputs[1]->format) {
        av_log(ctx, AV_LOG_ERROR, "Inputs must be of same pixel format.\n");
        return AVERROR(EINVAL);
    }

    s->desc = av_pix_fmt_desc_get(inlink->format);
    s->width = ctx->inputs[0]->w;
    s->height = ctx->inputs[0]->h;

    stride = ALIGN_CEIL(s->width * sizeof(float));
    data_sz = (size_t)stride * s->height;

    if (!(s->ref_data = av_malloc(data_sz))) {
        av_log(ctx, AV_LOG_ERROR, "ref data allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    if (!(s->main_data = av_malloc(data_sz))) {
        av_log(ctx, AV_LOG_ERROR, "main data allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    adm_buf_stride = ALIGN_CEIL(((s->width + 1) / 2) * sizeof(float));
    adm_buf_sz = (size_t)adm_buf_stride * ((s->height + 1) / 2);

    if (SIZE_MAX / adm_buf_sz < 35) {
        av_log(ctx, AV_LOG_ERROR, "error: SIZE_MAX / buf_sz_one < 35.\n");
        return AVERROR(EINVAL);
    }

    if (!(s->adm_data_buf = av_malloc(adm_buf_sz * 35))) {
        av_log(ctx, AV_LOG_ERROR, "data_buf allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    if (!(s->adm_temp_lo = av_malloc(stride))) {
        av_log(ctx, AV_LOG_ERROR, "adm temp lo allocation failed.\n");
        return AVERROR(ENOMEM);
    }
    if (!(s->adm_temp_hi = av_malloc(stride))) {
        av_log(ctx, AV_LOG_ERROR, "adm temp hi allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    if (!(s->prev_blur_data = av_mallocz(data_sz))) {
        av_log(ctx, AV_LOG_ERROR, "prev blur data allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    if (!(s->blur_data = av_mallocz(data_sz))) {
        av_log(ctx, AV_LOG_ERROR, "blur data allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    if (!(s->temp_data = av_mallocz(data_sz * 2))) {
        av_log(ctx, AV_LOG_ERROR, "temp data allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    if (SIZE_MAX / data_sz < 15) {
        av_log(ctx, AV_LOG_ERROR, "error: SIZE_MAX / buf_sz < 15\n");
        return AVERROR(EINVAL);
    }

    if (!(s->vif_data_buf = av_malloc(data_sz * 16)))
    {
        av_log(ctx, AV_LOG_ERROR, "error: av_malloc failed for data_buf.\n");
        return AVERROR(ENOMEM);
    }

    if (!(s->vif_temp = av_malloc(s->width * sizeof(float))))
    {
        av_log(ctx, AV_LOG_ERROR, "error: av_malloc failed for temp.\n");
        return AVERROR(ENOMEM);
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
    int i;

    if (s->nb_frames > 0) {
        append_array(&s->motion2_array, s->prev_motion_score);
        for(i = 0; i < s->nb_frames; i++) {
            printf("motion: %.3f motion2: %.3f\n",get_at_pos(&s->motion_array, i),get_at_pos(&s->motion2_array, i));
        }
                
        av_log(ctx, AV_LOG_INFO, "VMAF AVG: %.3f\n", s->vmaf_sum / s->nb_frames);        

        free_arr(&s->adm_array);
        for(i = 0; i < 4; i++) {
            free_arr(&s->adm_scale_array[i]);
        }
        free_arr(&s->motion_array);
        free_arr(&s->motion2_array);
        for(i = 0; i < 4; i++) {
            free_arr(&s->vif_scale_array[i]);
        }
        free_arr(&s->vif_array);
        
        av_free(s->ref_data);
        av_free(s->main_data);
        av_free(s->adm_data_buf);
        av_free(s->adm_temp_lo);
        av_free(s->adm_temp_hi);
        av_free(s->prev_blur_data);
        av_free(s->blur_data);
        av_free(s->temp_data);
        av_free(s->vif_data_buf);
        av_free(s->vif_temp);
    
    }

    ff_dualinput_uninit(&s->dinput);
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
