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
 * Calculate Anti-Noise Singnal to Noise Ratio (ANSNR) between two input videos.
 */

#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "dualinput.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "ansnr.h"
#include "video.h"

typedef struct ANSNRContext {
    const AVClass *class;
    FFDualInputContext dinput;
    const AVPixFmtDescriptor *desc;
    int width;
    int height;
    float *data_buf;
    double ansnr_sum;
    uint64_t nb_frames;
} ANSNRContext;

#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define OPT_RANGE_PIXEL_OFFSET (-128)

const int ansnr_filter2d_ref_width = 3;
const int ansnr_filter2d_main_width = 5;

const float ansnr_filter2d_ref[3 * 3] = {
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
    2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0
};
const float ansnr_filter2d_main[5 * 5] = {
    2.0 / 571.0,  7.0 / 571.0,  12.0 / 571.0,  7.0 / 571.0,  2.0 / 571.0,
    7.0 / 571.0, 31.0 / 571.0,  52.0 / 571.0, 31.0 / 571.0,  7.0 / 571.0,
    12.0 / 571.0, 52.0 / 571.0, 127.0 / 571.0, 52.0 / 571.0, 12.0 / 571.0,
    7.0 / 571.0, 31.0 / 571.0,  52.0 / 571.0, 31.0 / 571.0,  7.0 / 571.0,
    2.0 / 571.0,  7.0 / 571.0,  12.0 / 571.0,  7.0 / 571.0,  2.0 / 571.0
};

static const AVOption ansnr_options[] = {
    { NULL }
};

AVFILTER_DEFINE_CLASS(ansnr);

static inline float pow_2(float base)
{
    return base*base;
}

static void ansnr_mse(float *ref, float *main, float *signal, float *noise,
                      int w, int h, int ref_stride, int main_stride)
{
    int i, j;

    int ref_ind;
    int main_ind;

    float signal_sum = 0;
    float noise_sum = 0;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            ref_ind = i * ref_stride + j;
            main_ind = i * main_stride + j;

            signal_sum += pow_2(ref[ref_ind]);
            noise_sum += pow_2(ref[ref_ind] - main[main_ind]);
        }
    }

    if (signal) {
        *signal = signal_sum;
    }
    if (noise) {
        *noise = noise_sum;
    }
}

#define ansnr_filter2d_fn(type, bits) \
    static void ansnr_filter2d_##bits##bit(const float *filt, const uint8_t *source, float *dst, \
                                           int w, int h, int src_stride, int dst_stride, \
                                           int filt_width, ANSNRContext *s) \
{ \
    uint8_t sz; \
    \
    const type *src = (const type *) source; \
    \
    int src_px_stride; \
    \
    float filt_coeff, img_coeff; \
    int i, j, filt_i, filt_j, src_i, src_j; \
    \
    if (bits == 8) { \
        sz = sizeof(uint8_t); \
    } else { \
        sz = sizeof(uint16_t); \
    } \
    \
    src_px_stride = src_stride / sz; \
    \
    for (i = 0; i < h; i++) { \
        for (j = 0; j < w; j++) { \
            float accum = 0; \
            for (filt_i = 0; filt_i < filt_width; filt_i++) { \
                for (filt_j = 0; filt_j < filt_width; filt_j++) { \
                    filt_coeff = filt[filt_i * filt_width + filt_j]; \
                    \
                    src_i = i - filt_width / 2 + filt_i; \
                    src_j = j - filt_width / 2 + filt_j; \
                    \
                    src_i = FFABS(src_i); \
                    if (src_i >= h) { \
                        src_i = 2 * h - src_i - 1; \
                    } \
                    src_j = FFABS(src_j); \
                    if (src_j >= w) { \
                        src_j = 2 * w - src_j - 1; \
                    } \
                    \
                    img_coeff = src[src_i * src_px_stride + src_j] + \
                    OPT_RANGE_PIXEL_OFFSET; \
                    \
                    accum += filt_coeff * img_coeff; \
                } \
            } \
            dst[i * dst_stride + j] = accum; \
        } \
    } \
}

ansnr_filter2d_fn(uint8_t, 8);
ansnr_filter2d_fn(uint16_t, 10);

int compute_ansnr(const uint8_t *ref, const uint8_t *main, int w, int h,
                  int ref_stride, int main_stride, double *score,
                  double *score_psnr, double peak, double psnr_max, void *ctx)
{
    ANSNRContext *s = (ANSNRContext *) ctx;

    float *data_top;

    float *ref_filt;
    float *main_filt;

    float signal, noise;

    int buf_stride = ALIGN_CEIL(w * sizeof(float));
    size_t buf_sz = (size_t) (buf_stride * h);

    double eps = 1e-10;

    data_top = (float *) (s->data_buf);

    ref_filt = (float *) data_top;
    data_top += buf_sz / sizeof(float);
    main_filt = (float *) data_top;
    data_top += buf_sz / sizeof(float);

    buf_stride = buf_stride / sizeof(float);

    if (s->desc->comp[0].depth <= 8) {
        ansnr_filter2d_8bit(ansnr_filter2d_ref, (const uint8_t *) ref, ref_filt,
                            w, h, ref_stride, buf_stride,
                            ansnr_filter2d_ref_width, s);
        ansnr_filter2d_8bit(ansnr_filter2d_main, (const uint8_t *) main, main_filt,
                            w, h, main_stride, buf_stride,
                            ansnr_filter2d_main_width, s);
    } else {
        ansnr_filter2d_10bit(ansnr_filter2d_ref, (const uint8_t *) ref, ref_filt,
                             w, h, ref_stride, buf_stride,
                             ansnr_filter2d_ref_width, s);
        ansnr_filter2d_10bit(ansnr_filter2d_main, (const uint8_t *) main, main_filt,
                             w, h, main_stride, buf_stride,
                             ansnr_filter2d_main_width, s);
    }

    ansnr_mse(ref_filt, main_filt, &signal, &noise, w, h, buf_stride,
              buf_stride);

    *score = (noise==0) ? (psnr_max) : (10.0 * log10(signal / noise));

    *score_psnr = FFMIN(10.0 * log10(pow_2(peak) * w * h / FFMAX(noise, eps)),
                        psnr_max);

    return 0;
}

static void set_meta(AVDictionary **metadata, const char *key, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%0.2f", d);
    av_dict_set(metadata, key, value, 0);
}

static AVFrame *do_ansnr(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    ANSNRContext *s = ctx->priv;
    AVDictionary **metadata = &main->metadata;

    double score = 0.0;
    double score_psnr = 0.0;

    int w = s->width;
    int h = s->height;

    double stride;

    double max_psnr;
    double peak;

    uint8_t sz;

    if (s->desc->comp[0].depth <= 8) {
        peak = 255.0;
        max_psnr = 60.0;
        sz = sizeof(uint8_t);
    } else {
        peak = 255.75;
        max_psnr = 72.0;
        sz = sizeof(uint16_t);
    }

    stride = ALIGN_CEIL(w * sz);

    compute_ansnr((const uint8_t *) ref->data[0], (const uint8_t *) main->data[0],
                  w, h, stride, stride, &score, &score_psnr, peak, max_psnr, s);

    set_meta(metadata, "lavfi.ansnr.score", score);

    s->nb_frames++;

    s->ansnr_sum += score;

    return main;
}

static av_cold int init(AVFilterContext *ctx)
{
    ANSNRContext *s = ctx->priv;

    s->dinput.process = do_ansnr;

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
    ANSNRContext *s = ctx->priv;

    int buf_stride;
    size_t buf_sz;

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

    buf_stride = ALIGN_CEIL(s->width * sizeof(float));
    buf_sz = (size_t)buf_stride * s->height;

    if (SIZE_MAX / buf_sz < 3) {
        av_log(ctx, AV_LOG_ERROR, "SIZE_MAX / buf_sz < 3.\n");
        return AVERROR(EINVAL);
    }

    if (!(s->data_buf = av_malloc(buf_sz * 3))) {
        av_log(ctx, AV_LOG_ERROR, "data_buf allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    return 0;
}


static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    ANSNRContext *s = ctx->priv;
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
    ANSNRContext *s = inlink->dst->priv;
    return ff_dualinput_filter_frame(&s->dinput, inlink, inpicref);
}

static int request_frame(AVFilterLink *outlink)
{
    ANSNRContext *s = outlink->src->priv;
    return ff_dualinput_request_frame(&s->dinput, outlink);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    ANSNRContext *s = ctx->priv;

    av_free(s->data_buf);

    if(s->nb_frames > 0) {
        av_log(ctx, AV_LOG_INFO, "ANSNR AVG: %.3f\n", s->ansnr_sum / s->nb_frames);
    }

    ff_dualinput_uninit(&s->dinput);
}

static const AVFilterPad ansnr_inputs[] = {
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

static const AVFilterPad ansnr_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
        .request_frame = request_frame,
    },
    { NULL }
};

AVFilter ff_vf_ansnr = {
    .name          = "ansnr",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the ANSNR between two video streams."),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(ANSNRContext),
    .priv_class    = &ansnr_class,
    .inputs        = ansnr_inputs,
    .outputs       = ansnr_outputs,
};
