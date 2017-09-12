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
 * Calculate VMAF Motion score between two input videos.
 */

#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "drawutils.h"
#include "formats.h"
#include "framesync2.h"
#include "internal.h"
#include "vmaf_motion.h"
#include "video.h"

typedef struct VMAFMotionContext {
    const AVClass *class;
    FFFrameSync fs;
    const AVPixFmtDescriptor *desc;
    int conv_filter[5];
    int width;
    int height;
    uint16_t *prev_blur_data;
    uint16_t *blur_data;
    uint16_t *temp_data;
    double motion_sum;
    uint64_t nb_frames;
    VMAFMotionDSPContext dsp;
} VMAFMotionContext;

#define MAX_ALIGN 32
#define vmafmotion_options NULL
#define BIT_SHIFT 10

static const float FILTER_5[5] = {
    0.054488685,
    0.244201342,
    0.402619947,
    0.244201342,
    0.054488685
};

FRAMESYNC_DEFINE_CLASS(vmafmotion, VMAFMotionContext, fs);

static uint64_t image_sad(const uint16_t *img1, const uint16_t *img2, int w,
                          int h, ptrdiff_t img1_stride, ptrdiff_t img2_stride)
{
    uint64_t sum = 0;
    int i, j;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            sum += abs(img1[j] - img2[j]);
        }
        img1 += img1_stride / sizeof(uint16_t);
        img2 += img2_stride / sizeof(uint16_t);
    }

    return sum;
}

static void convolution_x(const int *filter, int filt_w, const uint16_t *src,
                          uint16_t *dst, int w, int h, ptrdiff_t src_stride,
                          ptrdiff_t dst_stride)
{
    int radius = filt_w / 2;
    int borders_left = radius;
    int borders_right = w - (filt_w - radius);
    int i, j, k;
    int sum = 0;

    for (i = 0; i < h; i++) {
        for (j = 0; j < borders_left; j++) {
            sum = 0;
            for (k = 0; k < filt_w; k++) {
                int j_tap = FFABS(j - radius + k);
                if (j_tap >= w) {
                    j_tap = w - (j_tap - w + 1);
                }
                sum += filter[k] * src[i * src_stride / sizeof(uint16_t) + j_tap];
            }
            dst[i * dst_stride / sizeof(uint16_t) + j] = sum >> BIT_SHIFT;
        }

        for (j = borders_left; j < borders_right; j++) {
            int sum = 0;
            for (k = 0; k < filt_w; k++) {
                sum += filter[k] * src[i * src_stride / sizeof(uint16_t) + j - radius + k];
            }
            dst[i * dst_stride / sizeof(uint16_t) + j] = sum >> BIT_SHIFT;
        }

        for (j = borders_right; j < w; j++) {
            sum = 0;
            for (k = 0; k < filt_w; k++) {
                int j_tap = FFABS(j - radius + k);
                if (j_tap >= w) {
                    j_tap = w - (j_tap - w + 1);
                }
                sum += filter[k] * src[i * src_stride / sizeof(uint16_t) + j_tap];
            }
            dst[i * dst_stride / sizeof(uint16_t) + j] = sum >> BIT_SHIFT;
        }
    }
}

#define conv_y_fn(type, bits) \
    static void convolution_y_##bits##bit(const int *filter, int filt_w, \
                                          const type *src, uint16_t *dst, \
                                          int w, int h, ptrdiff_t src_stride, \
                                          ptrdiff_t dst_stride) \
{ \
    int radius = filt_w / 2; \
    int borders_top = radius; \
    int borders_bottom = h - (filt_w - radius); \
    int i, j, k; \
    int sum = 0; \
    \
    for (i = 0; i < borders_top; i++) { \
        for (j = 0; j < w; j++) { \
            sum = 0; \
            for (k = 0; k < filt_w; k++) { \
                int i_tap = FFABS(i - radius + k); \
                if (i_tap >= h) { \
                    i_tap = h - (i_tap - h + 1); \
                } \
                sum += filter[k] * src[i_tap * src_stride / sizeof(type) + j]; \
            } \
            dst[i * dst_stride / sizeof(uint16_t) + j] = sum >> BIT_SHIFT; \
        } \
    } \
    for (i = borders_top; i < borders_bottom; i++) { \
        for (j = 0; j < w; j++) { \
            sum = 0; \
            for (k = 0; k < filt_w; k++) { \
                sum += filter[k] * src[(i - radius + k) * src_stride / sizeof(type) + j]; \
            } \
            dst[i * dst_stride / sizeof(uint16_t) + j] = sum >> BIT_SHIFT; \
        } \
    } \
    for (i = borders_bottom; i < h; i++) { \
        for (j = 0; j < w; j++) { \
            sum = 0; \
            for (k = 0; k < filt_w; k++) { \
                int i_tap = FFABS(i - radius + k); \
                if (i_tap >= h) { \
                    i_tap = h - (i_tap - h + 1); \
                } \
                sum += filter[k] * src[i_tap * src_stride / sizeof(type) + j]; \
            } \
            dst[i * dst_stride / sizeof(uint16_t) + j] = sum >> BIT_SHIFT; \
        } \
    } \
}

conv_y_fn(uint8_t, 8);
conv_y_fn(uint16_t, 10);

int compute_vmafmotion(const int *filter, int filt_w, void *src,
                       uint16_t *temp_data, const uint16_t *ref, uint16_t *main,
                       int w, int h, ptrdiff_t ref_stride, ptrdiff_t main_stride,
                       double *score,  ptrdiff_t src_stride, uint8_t bitdepth)
{
    uint64_t sad;
    if(bitdepth <= 8) {
        convolution_y_8bit(filter, 5, (const uint8_t *) src, temp_data, w, h,
                           src_stride, ref_stride);
    } else {
        convolution_y_10bit(filter, 5, (const uint16_t *) src, temp_data, w, h,
                            src_stride, ref_stride);
    }

    convolution_x(filter, 5, temp_data, main, w, h, ref_stride, ref_stride);

    sad = image_sad(ref, main, w, h, ref_stride, main_stride);
    *score = (double) (sad * 1.0 / (w * h));

    return 0;
}

static void set_meta(AVDictionary **metadata, const char *key, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%0.2f", d);
    av_dict_set(metadata, key, value, 0);
}

static int do_vmafmotion(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    VMAFMotionContext *s = ctx->priv;
    AVFrame *main, *ref;
    AVDictionary **metadata;
    int ret;
    ptrdiff_t ref_stride;
    ptrdiff_t stride;

    double score;

    ret = ff_framesync2_dualinput_get(fs, &main, &ref);
    if (ret < 0)
        return ret;
    if (!ref)
        return ff_filter_frame(ctx->outputs[0], main);

    metadata = &main->metadata;

    ref_stride = ref->linesize[0];
    stride = FFALIGN(s->width * sizeof(uint16_t), MAX_ALIGN);

    compute_vmafmotion(s->conv_filter, 5, ref->data[0], s->temp_data,
                       s->prev_blur_data, s->blur_data, s->width, s->height,
                       stride, stride, &score, ref_stride, s->desc->comp[0].depth);

    if(!s->nb_frames) {
        score = 0.0;
    }

    FFSWAP(uint16_t *, s->prev_blur_data, s->blur_data);

    set_meta(metadata, "lavfi.vmafmotion.score", score);

    s->nb_frames++;

    s->motion_sum += score;

    return ff_filter_frame(ctx->outputs[0], main);
}

static av_cold int init(AVFilterContext *ctx)
{
    VMAFMotionContext *s = ctx->priv;

    int i;
    for(i = 0; i < 5; i++) {
        s->conv_filter[i] = lrint(FILTER_5[i] * (1 << BIT_SHIFT));
    }

    s->fs.on_event = do_vmafmotion;

    return 0;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_YUV444P10, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV420P10,
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
    VMAFMotionContext *s = ctx->priv;
    ptrdiff_t stride;
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

    s->desc = av_pix_fmt_desc_get(inlink->format);
    s->width = ctx->inputs[0]->w;
    s->height = ctx->inputs[0]->h;

    stride = FFALIGN(s->width * sizeof(uint16_t), MAX_ALIGN);
    data_sz = (size_t)stride * s->height;

    if (!(s->prev_blur_data = av_malloc(data_sz))) {
        return AVERROR(ENOMEM);
    }
    if (!(s->blur_data = av_malloc(data_sz))) {
        return AVERROR(ENOMEM);
    }
    if (!(s->temp_data = av_malloc(data_sz))) {
        return AVERROR(ENOMEM);
    }

    s->dsp.image_sad = image_sad;
    if (ARCH_X86)
        ff_vmafmotion_init_x86(&s->dsp);

    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    VMAFMotionContext *s = ctx->priv;
    AVFilterLink *mainlink = ctx->inputs[0];
    int ret;

    ret = ff_framesync2_init_dualinput(&s->fs, ctx);
    if (ret < 0)
        return ret;
    outlink->w = mainlink->w;
    outlink->h = mainlink->h;
    outlink->time_base = mainlink->time_base;
    outlink->sample_aspect_ratio = mainlink->sample_aspect_ratio;
    outlink->frame_rate = mainlink->frame_rate;
    if ((ret = ff_framesync2_configure(&s->fs)) < 0)
        return ret;
    return 0;
}

static int activate(AVFilterContext *ctx)
{
    VMAFMotionContext *s = ctx->priv;
    return ff_framesync2_activate(&s->fs);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    VMAFMotionContext *s = ctx->priv;

    ff_framesync2_uninit(&s->fs);

    if (s->nb_frames > 0) {
        av_log(ctx, AV_LOG_INFO, "VMAF Motion avg: %.3f\n", s->motion_sum / s->nb_frames);
    }

    av_free(s->prev_blur_data);
    av_free(s->blur_data);
    av_free(s->temp_data);
}

static const AVFilterPad vmafmotion_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
    },{
        .name         = "reference",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input_ref,
    },
    { NULL }
};

static const AVFilterPad vmafmotion_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_vf_vmafmotion = {
    .name          = "vmafmotion",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the VMAF Motion score between two video streams."),
    .preinit       = vmafmotion_framesync_preinit,
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .activate      = activate,
    .priv_size     = sizeof(VMAFMotionContext),
    .priv_class    = &vmafmotion_class,
    .inputs        = vmafmotion_inputs,
    .outputs       = vmafmotion_outputs,
};
