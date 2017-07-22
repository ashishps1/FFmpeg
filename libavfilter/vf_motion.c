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

typedef struct MotionContext {
    const AVClass *class;
    FFDualInputContext dinput;
    const AVPixFmtDescriptor *desc;
    int width;
    int height;
    float *ref_data;
    float *prev_blur_data;
    float *blur_data;
    float *temp_data;
    double motion_sum;
    uint64_t nb_frames;
} MotionContext;

#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define OPT_RANGE_PIXEL_OFFSET (-128)

static const AVOption motion_options[] = {
    { NULL }
};

AVFILTER_DEFINE_CLASS(motion);

static float image_sad(const float *img1, const float *img2, int w,
                          int h, int img1_stride, int img2_stride)
{
    float accum = 0.0;

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float img1px = img1[i * img1_stride + j];
            float img2px = img2[i * img2_stride + j];

            accum += fabs(img1px - img2px);
        }
    }

    return (float) (accum / (w * h));
}

int compute_motion1(const float *ref, const float *dis, int w, int h,
                          int ref_stride, int dis_stride, double *score,
                          void *ctx)
{
    *score = image_sad(ref, dis, w, h, ref_stride / sizeof(float),
                         dis_stride / sizeof(float));

    return 0;
}

static inline int floorn(int n, int m)
{
    return n - n % m;
}

static inline int ceiln(int n, int m)
{
    return n % m ? n + (m - n % m) : n;
}

av_always_inline static float convolution_edge(int horizontal, const float *filter,
                                        int filt_width, const float *src,
                                        int w, int h, int stride, int i,
                                        int j)
{
    int radius = filt_width / 2;

    float accum = 0;
    for (int k = 0; k < filt_width; ++k) {
        int i_tap = horizontal ? i : i - radius + k;
        int j_tap = horizontal ? j - radius + k : j;

        if (horizontal) {
            j_tap = FFABS(j_tap);
            if (j_tap >= w) {
                j_tap = w - (j_tap - w + 1);
            }
        } else {
            i_tap = FFABS(i_tap);
            if (i_tap >= h)
                i_tap = h - (i_tap - h + 1);
        }

        accum += filter[k] * src[i_tap * stride + j_tap];
    }
    return accum;
}

static void convolution_x(const float *filter, int filt_width,
                            const float *src, float *dst, int w, int h,
                            int src_stride, int dst_stride, int step)
{
    int radius = filt_width / 2;
    int borders_left = ceiln(radius, step);
    int borders_right = floorn(w - (filt_width - radius), step);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < borders_left; j += step) {
            dst[i * dst_stride + j / step] = convolution_edge(1, filter,
                                                              filt_width, src,
                                                              w, h, src_stride,
                                                              i, j);
        }

        for (int j = borders_left; j < borders_right; j += step) {
            float accum = 0;
            for (int k = 0; k < filt_width; k++) {
                accum += filter[k] * src[i * src_stride + j - radius + k];
            }
            dst[i * dst_stride + j / step] = accum;
        }

        for (int j = borders_right; j < w; j += step) {
            dst[i * dst_stride + j / step] = convolution_edge(1, filter,
                                                              filt_width, src,
                                                              w, h, src_stride,
                                                              i, j);
        }
    }
}

static void convolution_y(const float *filter, int filt_width,
                            const float *src, float *dst, int w, int h,
                            int src_stride, int dst_stride, int step)
{
    int radius = filt_width / 2;
    int borders_top = ceiln(radius, step);
    int borders_bottom = floorn(h - (filt_width - radius), step);

    for (int i = 0; i < borders_top; i += step) {
        for (int j = 0; j < w; j++) {
            dst[(i / step) * dst_stride + j] = convolution_edge(0, filter,
                                                                filt_width, src,
                                                                w, h, src_stride,
                                                                i, j);
        }
    }
    for (int i = borders_top; i < borders_bottom; i += step) {
        for (int j = 0; j < w; j++) {
            float accum = 0;
            for (int k = 0; k < filt_width; k++) {
                accum += filter[k] * src[(i - radius + k) * src_stride + j];
            }
            dst[(i / step) * dst_stride + j] = accum;
        }
    }
    for (int i = borders_bottom; i < h; i += step) {
        for (int j = 0; j < w; j++) {
            dst[(i / step) * dst_stride + j] = convolution_edge(0, filter,
                                                                filt_width, src,
                                                                w, h, src_stride,
                                                                i, j);
        }
    }
}

static void convolution_f32(const float *filter, int filt_width, const float *src,
                       float *dst, float *tmp, int w, int h, int src_stride,
                       int dst_stride)
{
    convolution_y(filter, filt_width, src, tmp, w, h, src_stride,
                    dst_stride, 1);
    convolution_x(filter, filt_width, tmp, dst, w, h, src_stride,
                    dst_stride, 1);
}

#define offset_fn(type, bits) \
    static void offset_##bits##bit(MotionContext *s, const AVFrame *ref, int stride) \
{ \
    int w = s->width; \
    int h = s->height; \
    int i,j; \
    \
    int ref_stride = ref->linesize[0]; \
    \
    const type *ref_ptr = (const type *) ref->data[0]; \
    \
    float *ref_ptr_data = s->ref_data; \
    \
    for(i = 0; i < h; i++) { \
        for(j = 0; j < w; j++) { \
            ref_ptr_data[j] = (float) ref_ptr[j] + OPT_RANGE_PIXEL_OFFSET; \
        } \
        ref_ptr += ref_stride / sizeof(type); \
        ref_ptr_data += stride / sizeof(float); \
    } \
}

offset_fn(uint8_t, 8);
offset_fn(uint16_t, 10);

static void set_meta(AVDictionary **metadata, const char *key, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%0.2f", d);
    av_dict_set(metadata, key, value, 0);
}

static AVFrame *do_motion(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    MotionContext *s = ctx->priv;
    AVDictionary **metadata = &main->metadata;
    int stride;
    size_t data_sz;
    double score;

    stride = ALIGN_CEIL(s->width * sizeof(float));
    data_sz = (size_t)stride * s->height;

    /** Offset ref and main pixel by OPT_RANGE_PIXEL_OFFSET */
    if (s->desc->comp[0].depth <= 8) {
        offset_8bit(s, ref, stride);
    } else {
        offset_10bit(s, ref, stride);
    }

    convolution_f32(FILTER_5, 5, s->ref_data, s->blur_data, s->temp_data,
                      s->width, s->height, stride / sizeof(float), stride /
                      sizeof(float));

    if(!s->nb_frames) {
        score = 0.0;
    } else {
        compute_motion1(s->prev_blur_data, s->blur_data, s->width, s->height,
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
    MotionContext *s = ctx->priv;

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
    AVFilterContext *ctx  = inlink->dst;
    MotionContext *s = ctx->priv;
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

    s->desc = av_pix_fmt_desc_get(inlink->format);
    s->width = ctx->inputs[0]->w;
    s->height = ctx->inputs[0]->h;

    stride = ALIGN_CEIL(s->width * sizeof(float));
    data_sz = (size_t)stride * s->height;

    if (!(s->ref_data = av_malloc(data_sz))) {
        av_log(ctx, AV_LOG_ERROR, "ref_buf allocation failed.\n");
        return AVERROR(ENOMEM);
    }
    if (!(s->prev_blur_data = av_mallocz(data_sz))) {
        av_log(ctx, AV_LOG_ERROR, "prev_blur_buf allocation failed.\n");
        return AVERROR(ENOMEM);
    }
    if (!(s->blur_data = av_mallocz(data_sz))) {
        av_log(ctx, AV_LOG_ERROR, "blur_buf allocation failed.\n");
        return AVERROR(ENOMEM);
    }
    if (!(s->temp_data = av_mallocz(data_sz * 2))) {
        av_log(ctx, AV_LOG_ERROR, "temp_buf allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    MotionContext *s = ctx->priv;
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
    MotionContext *s = inlink->dst->priv;
    return ff_dualinput_filter_frame(&s->dinput, inlink, inpicref);
}

static int request_frame(AVFilterLink *outlink)
{
    MotionContext *s = outlink->src->priv;
    return ff_dualinput_request_frame(&s->dinput, outlink);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    MotionContext *s = ctx->priv;

    if (s->nb_frames > 0) {
        av_log(ctx, AV_LOG_INFO, "Motion AVG: %.3f\n", s->motion_sum / s->nb_frames);
    }

    av_free(s->ref_data);
    av_free(s->prev_blur_data);
    av_free(s->blur_data);
    av_free(s->temp_data);

    ff_dualinput_uninit(&s->dinput);
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
    .description   = NULL_IF_CONFIG_SMALL("Calculate the Motion between two video streams."),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(MotionContext),
    .priv_class    = &motion_class,
    .inputs        = motion_inputs,
    .outputs       = motion_outputs,
};
