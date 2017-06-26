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
 * Caculate the ANSNR between two input videos.
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
#include "video.h"

typedef struct ANSNRContext {
    const AVClass *class;
    FFDualInputContext dinput;
    double ansnr_sum;
    uint64_t nb_frames;
    int nb_components;
} ANSNRContext;

#define OFFSET(x) offsetof(ANSNRContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM
#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))

static const AVOption ansnr_options[] = {
    { NULL }
};

AVFILTER_DEFINE_CLASS(ansnr);


const float ansnr_filter1d_ref_s[3] = { 0x1.00243ap-2, 0x1.ffb78cp-2, 0x1.00243ap-2 };

const float ansnr_filter1d_dis_s[5] = { 0x1.be5f0ep-5, 0x1.f41fd6p-3, 0x1.9c4868p-2, 0x1.f41fd6p-3, 0x1.be5f0ep-5 };

const int ansnr_filter1d_ref_width = 3;
const int ansnr_filter1d_dis_width = 5;

const float ansnr_filter2d_ref_s[3*3] = {
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
    2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0
};

const float ansnr_filter2d_dis_s[5*5] = {
    2.0 / 571.0,  7.0 / 571.0,  12.0 / 571.0,  7.0 / 571.0,  2.0 / 571.0,
    7.0 / 571.0,  31.0 / 571.0, 52.0 / 571.0,  31.0 / 571.0, 7.0 / 571.0,
    12.0 / 571.0, 52.0 / 571.0, 127.0 / 571.0, 52.0 / 571.0, 12.0 / 571.0,
    7.0 / 571.0,  31.0 / 571.0, 52.0 / 571.0,  31.0 / 571.0, 7.0 / 571.0,
    2.0 / 571.0,  7.0 / 571.0,  12.0 / 571.0,  7.0 / 571.0,  2.0 / 571.0
};

const int ansnr_filter2d_ref_width = 3;
const int ansnr_filter2d_dis_width = 5;


static void ansnr_mse(const uint8_t *ref, const uint8_t *dis, float *sig,
                      float *noise, int w, int h, int ref_stride,
                      int dis_stride)
{
    int ref_px_stride = ref_stride / sizeof(uint8_t);
    int dis_px_stride = dis_stride / sizeof(uint8_t);
    int i, j;

    float ref_val, dis_val;

    float sig_accum = 0;
    float noise_accum = 0;

    for (i = 0; i < h; ++i) {
        float sig_accum_inner = 0;
        float noise_accum_inner = 0;

        for (j = 0; j < w; ++j) {
            ref_val = ref[i * ref_px_stride + j];
            dis_val = dis[i * dis_px_stride + j];

            sig_accum_inner   += ref_val * ref_val;
            noise_accum_inner += (ref_val - dis_val) * (ref_val - dis_val);
        }

        sig_accum   += sig_accum_inner;
        noise_accum += noise_accum_inner;
    }

    if (sig)
        *sig = sig_accum;
    if (noise)
        *noise = noise_accum;
}

static void ansnr_filter1d(const uint8_t *f, const uint8_t *src, uint8_t *dst,
                           int w, int h, int src_stride, int dst_stride,
                           int fwidth)
{
    int src_px_stride = src_stride / sizeof(uint8_t);
    int dst_px_stride = dst_stride / sizeof(uint8_t);

    float *tmp = aligned_malloc(ALIGN_CEIL(w * sizeof(float)), MAX_ALIGN);
    float fcoeff, imgcoeff;

    int i, j, fi, fj, ii, jj;

    for (i = 0; i < h; ++i) {
        /* Vertical pass. */
        for (j = 0; j < w; ++j) {
            float accum = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                fcoeff = f[fi];

                ii = i - fwidth / 2 + fi;
#ifdef ANSNR_OPT_BORDER_REPLICATE
                ii = ii < 0 ? 0 : (ii > h - 1 ? h - 1 : ii);
                imgcoeff = src[ii * src_px_stride + j];
#else
                if (ii < 0) ii = -ii;
                else if (ii >= h) ii = 2 * h - ii - 1;
                imgcoeff = src[ii * src_px_stride + j];
#endif
                accum += fcoeff * imgcoeff;
            }

            tmp[j] = accum;
        }

        /* Horizontal pass. */
        for (j = 0; j < w; ++j) {
            float accum = 0;

            for (fj = 0; fj < fwidth; ++fj) {
                fcoeff = f[fj];

                jj = j - fwidth / 2 + fj;
#ifdef ANSNR_OPT_BORDER_REPLICATE
                jj = jj < 0 ? 0 : (jj > w - 1 ? w - 1 : jj);
                imgcoeff = tmp[jj];
#else
                if (jj < 0) jj = -jj;
                else if (jj >= w) jj = 2 * w - jj - 1;
                imgcoeff = tmp[jj];
#endif
                accum += fcoeff * imgcoeff;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }

    aligned_free(tmp);
}

static void ansnr_filter2d(const uint8_t *f, const uint8_t *src, uint8_t *dst,
                           int w, int h, int src_stride, int dst_stride,
                           int fwidth)
{
    int src_px_stride = src_stride / sizeof(uint8_t);
    int dst_px_stride = dst_stride / sizeof(uint8_t);

    float fcoeff, imgcoeff;
    int i, j, fi, fj, ii, jj;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            float accum = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                float accum_inner = 0;

                for (fj = 0; fj < fwidth; ++fj) {
                    fcoeff = f[fi * fwidth + fj];

                    ii = i - fwidth / 2 + fi;
                    jj = j - fwidth / 2 + fj;
#ifdef ANSNR_OPT_BORDER_REPLICATE
                    ii = ii < 0 ? 0 : (ii > h - 1 ? h - 1 : ii);
                    jj = jj < 0 ? 0 : (jj > w - 1 ? w - 1 : jj);
                    imgcoeff = src[ii * src_px_stride + jj];
#else
                    if (ii < 0) ii = -ii;
                    else if (ii >= h) ii = 2 * h - ii - 1;
                    if (jj < 0) jj = -jj;
                    else if (jj >= w) jj = 2 * w - jj - 1;
                    imgcoeff = src[ii * src_px_stride + jj];
#endif
                    accum_inner += fcoeff * imgcoeff;
                }

                accum += accum_inner;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }
}

static int compute_ansnr(const uint8_t *ref, const number_t *dis, int w,
                         int h, int ref_stride, int dis_stride, double *score,
                         double *score_psnr, double peak, double psnr_max)
{
    uint8_t *data_buf = 0;
    char *data_top;

    uint8_t *ref_filtr;
    uint8_t *ref_filtd;
    uint8_t *dis_filtd;

    float sig, noise;

#ifdef ANSNR_OPT_NORMALIZE
    float noise_min;
#endif

    int buf_stride = ALIGN_CEIL(w * sizeof(uint8_t));
    size_t buf_sz_one = (size_t)buf_stride * h;

    int ret = 1;

    if (SIZE_MAX / buf_sz_one < 3)
    {
        goto fail;
    }

    if (!(data_buf = aligned_malloc(buf_sz_one * 3, MAX_ALIGN)))
    {
        goto fail;
    }

    data_top = (char *)data_buf;

    ref_filtr = (uint8_t *)data_top; data_top += buf_sz_one;
    ref_filtd = (uint8_t *)data_top; data_top += buf_sz_one;
    dis_filtd = (uint8_t *)data_top; data_top += buf_sz_one;

#ifdef ANSNR_OPT_FILTER_1D
    ansnr_filter1d(ansnr_filter1d_ref, ref, ref_filtr, w, h, ref_stride, buf_stride, ansnr_filter1d_ref_width);
    ansnr_filter1d(ansnr_filter1d_dis, ref, ref_filtd, w, h, ref_stride, buf_stride, ansnr_filter1d_dis_width);
    ansnr_filter1d(ansnr_filter1d_dis, dis, dis_filtd, w, h, dis_stride, buf_stride, ansnr_filter1d_dis_width);
#else
    ansnr_filter2d(ansnr_filter2d_ref, ref, ref_filtr, w, h, ref_stride, buf_stride, ansnr_filter2d_ref_width);
    ansnr_filter2d(ansnr_filter2d_dis, ref, ref_filtd, w, h, ref_stride, buf_stride, ansnr_filter2d_dis_width);
    ansnr_filter2d(ansnr_filter2d_dis, dis, dis_filtd, w, h, dis_stride, buf_stride, ansnr_filter2d_dis_width);
#endif

#ifdef ANSNR_OPT_DEBUG_DUMP
    write_image("stage/ref_filtr.bin", ref_filtr, w, h, buf_stride, sizeof(uint8_t));
    write_image("stage/ref_filtd.bin", ref_filtd, w, h, buf_stride, sizeof(uint8_t));
    write_image("stage/dis_filtd.bin", dis_filtd, w, h, buf_stride, sizeof(uint8_t));
#endif

    ansnr_mse(ref_filtr, dis_filtd, &sig, &noise, w, h, buf_stride, buf_stride);

#ifdef ANSNR_OPT_NORMALIZE
    ansnr_mse(ref_filtr, ref_filtd, 0, &noise_min, w, h, buf_stride, buf_stride);
    *score = 10.0 * log10(noise / (noise - noise_min));
#else
    *score = noise==0 ? psnr_max : 10.0 * log10(sig / noise);
#endif

    double eps = 1e-10;
    *score_psnr = MIN(10 * log10(peak * peak * w * h / MAX(noise, eps)), psnr_max);

    ret = 0;
fail:
    aligned_free(data_buf);
    return ret;
}

static AVFrame *do_ansnr(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    ANSNRContext *s = ctx->priv;

    char *format = (char *)av_get_pix_fmt_name(main->format);

    double max_psnr;

    if (!strcmp(format, "yuv420p") || !strcmp(format, "yuv422p") || !strcmp(format, "yuv444p")) {
        max_psnr = 60;
    }
    else if (!strcmp(format, "yuv420p10le") || !strcmp(format, "yuv422p10le") || !strcmp(format, "yuv444p10le")) {
        max_psnr = 72;
    }

    double score = 0.0;

    int w = s->planewidth[0];
    int h = s->planeheight[0];

    double stride;

    stride = ALIGN_CEIL(w * sizeof(uint8_t));

    double score_psnr;

    compute_ansnr(ref->data[0], main->data[0], s->planewidth[0], s->planeheight[0], stride, stride, &score, &score_psnr, 255.75, max_psnr);

    s->nb_frames++;

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
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    AVFilterContext *ctx  = inlink->dst;
    ANSNRContext *s = ctx->priv;
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

    ff_dualinput_uninit(&s->dinput);

    if (s->stats_file && s->stats_file != stdout)
        fclose(s->stats_file);
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
    .description   = NULL_IF_CONFIG_SMALL("Calculate the PSNR between two video streams."),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(ANSNRContext),
    .priv_class    = &ansnr_class,
    .inputs        = ansnr_inputs,
    .outputs       = ansnr_outputs,
};
