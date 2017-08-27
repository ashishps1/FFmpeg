
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
 * Calculate VIF between two input videos.
 */

#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "dualinput.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "vif.h"
#include "video.h"

typedef struct VIFContext {
    const AVClass *class;
    FFDualInputContext dinput;
    const AVPixFmtDescriptor *desc;
    int vif_filter[4][17];
    int width;
    int height;
    uint64_t *data_buf;
    uint64_t *temp;
    double vif_sum;
    uint64_t nb_frames;
} VIFContext;

#define OFFSET(x) offsetof(VIFContext, x)
#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define OPT_RANGE_PIXEL_OFFSET (-128)

static const AVOption vif_options[] = {
    { NULL }
};

AVFILTER_DEFINE_CLASS(vif);

static void vif_dec2(const uint64_t *src, uint64_t *dst, int src_w, int src_h,
                     ptrdiff_t src_stride, ptrdiff_t dst_stride)
{
    ptrdiff_t src_px_stride = src_stride / sizeof(uint64_t);
    ptrdiff_t dst_px_stride = dst_stride / sizeof(uint64_t);

    int i, j;

    /** decimation by 2 in each direction (after gaussian blur check) */
    for (i = 0; i < src_h / 2; i++) {
        for (j = 0; j < src_w / 2; j++) {
            dst[i * dst_px_stride + j] = src[(i * 2) * src_px_stride + (j * 2)];
        }
    }
}

static int vif_sum(const uint64_t *x, int w, int h, ptrdiff_t stride)
{
    ptrdiff_t px_stride = stride / sizeof(uint64_t);
    int i, j;

    int sum = 0;

    for (i = 0; i < h; i++) {
        int sum_inner = 0;

        for (j = 0; j < w; j++) {
            sum_inner += x[i * px_stride + j];
        }

        sum += sum_inner;
    }

    return sum;
}

static void vif_statistic(const uint64_t *mu1_sq, const uint64_t *mu2_sq,
                          const uint64_t *mu1_mu2, const uint64_t *xx_filt,
                          const uint64_t *yy_filt, const uint64_t *xy_filt,
                          uint64_t *num, uint64_t *den, int w, int h,
                          ptrdiff_t mu1_sq_stride, ptrdiff_t mu2_sq_stride,
                          ptrdiff_t mu1_mu2_stride, ptrdiff_t xx_filt_stride,
                          ptrdiff_t yy_filt_stride, ptrdiff_t xy_filt_stride,
                          ptrdiff_t num_stride, ptrdiff_t den_stride)
{
    static const float sigma_nsq = 2;
    static const float sigma_max_inv = 4.0 / (255.0 * 255.0);

    ptrdiff_t mu1_sq_px_stride  = mu1_sq_stride / sizeof(uint64_t);
    ptrdiff_t mu2_sq_px_stride  = mu2_sq_stride / sizeof(uint64_t);
    ptrdiff_t mu1_mu2_px_stride = mu1_mu2_stride / sizeof(uint64_t);
    ptrdiff_t xx_filt_px_stride = xx_filt_stride / sizeof(uint64_t);
    ptrdiff_t yy_filt_px_stride = yy_filt_stride / sizeof(uint64_t);
    ptrdiff_t xy_filt_px_stride = xy_filt_stride / sizeof(uint64_t);
    ptrdiff_t num_px_stride = num_stride / sizeof(uint64_t);
    ptrdiff_t den_px_stride = den_stride / sizeof(uint64_t);

    float mu1_sq_val, mu2_sq_val, mu1_mu2_val, xx_filt_val, yy_filt_val, xy_filt_val;
    float sigma1_sq, sigma2_sq, sigma12, g, sv_sq;
    float num_val, den_val;
    int i, j;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            mu1_sq_val  = mu1_sq[i * mu1_sq_px_stride + j];
            mu2_sq_val  = mu2_sq[i * mu2_sq_px_stride + j];
            mu1_mu2_val = mu1_mu2[i * mu1_mu2_px_stride + j];
            xx_filt_val = xx_filt[i * xx_filt_px_stride + j];
            yy_filt_val = yy_filt[i * yy_filt_px_stride + j];
            xy_filt_val = xy_filt[i * xy_filt_px_stride + j];

            sigma1_sq = xx_filt_val - mu1_sq_val;
            sigma2_sq = yy_filt_val - mu2_sq_val;
            sigma12 = xy_filt_val - mu1_mu2_val;

            if (sigma1_sq < sigma_nsq) {
                num_val = 1.0 - sigma2_sq * sigma_max_inv;
                den_val = 1.0;
            } else {
                sv_sq = (sigma2_sq + sigma_nsq) * sigma1_sq;
                if( sigma12 < 0 ) {
                    num_val = 0.0;
                } else {
                    g = sv_sq - sigma12 * sigma12;
                    num_val = log2f(1.0 + sv_sq / g);
                }
                den_val = log2f(1.0 + sigma1_sq / sigma_nsq);
            }

            num[i * num_px_stride + j] = num_val;
            den[i * den_px_stride + j] = den_val;
        }
    }
}

#define vif_xy_fn(type, bits) \
    static void vif_xx_yy_xy_##bits##bit(const type *x, const type *y, uint64_t *xx, uint64_t *yy, \
                                         uint64_t *xy, int w, int h, ptrdiff_t xstride, ptrdiff_t ystride, \
                                         ptrdiff_t xxstride, ptrdiff_t yystride, ptrdiff_t xystride) \
{ \
    ptrdiff_t x_px_stride = xstride / sizeof(type); \
    ptrdiff_t y_px_stride = ystride / sizeof(type); \
    ptrdiff_t xx_px_stride = xxstride / sizeof(uint64_t); \
    ptrdiff_t yy_px_stride = yystride / sizeof(uint64_t); \
    ptrdiff_t xy_px_stride = xystride / sizeof(uint64_t); \
    \
    int i, j; \
    \
    uint64_t xval, yval, xxval, yyval, xyval; \
    \
    for (i = 0; i < h; i++) { \
        for (j = 0; j < w; j++) { \
            xval = (uint64_t) x[i * x_px_stride + j]; \
            yval = (uint64_t) y[i * y_px_stride + j]; \
            \
            xxval = xval * xval; \
            yyval = yval * yval; \
            xyval = xval * yval; \
            \
            xx[i * xx_px_stride + j] = xxval; \
            yy[i * yy_px_stride + j] = yyval; \
            xy[i * xy_px_stride + j] = xyval; \
        } \
    } \
}

#define vif_filter1d_fn(type, bits) \
    static void vif_filter1d_##bits##bit(const int *filter, const type *src, uint64_t *dst, \
                                         uint64_t *temp_buf, int w, int h, ptrdiff_t src_stride, \
                                         ptrdiff_t dst_stride, int filt_w, uint64_t *temp) \
{ \
    ptrdiff_t src_px_stride = src_stride / sizeof(type); \
    ptrdiff_t dst_px_stride = dst_stride / sizeof(uint64_t); \
    \
    int i, j, filt_i, filt_j, ii, jj; \
    \
    for (i = 0; i < h; i++) { \
        /** Vertical pass. */ \
        for (j = 0; j < w; j++) { \
            uint64_t sum = 0; \
            \
            for (filt_i = 0; filt_i < filt_w; filt_i++) { \
                ii = i - filt_w / 2 + filt_i; \
                ii = FFABS(ii); \
                if(ii >= h) { \
                    ii = 2 * h - ii - 1; \
                } \
                \
                sum += filter[filt_i] * src[ii * src_px_stride + j]; \
            } \
            temp[j] = sum >> N; \
        } \
        \
        /** Horizontal pass. */ \
        for (j = 0; j < w; j++) { \
            uint64_t sum = 0; \
            \
            for (filt_j = 0; filt_j < filt_w; filt_j++) { \
                jj = j - filt_w / 2 + filt_j; \
                jj = FFABS(jj); \
                if(jj >= w) { \
                    jj = 2 * w - jj - 1; \
                } \
                \
                sum += filter[filt_j] * temp[jj]; \
            } \
            dst[i * dst_px_stride + j] = sum >> N; \
        } \
    } \
}

vif_filter1d_fn(uint8_t, 8);
vif_filter1d_fn(uint16_t, 16);
vif_filter1d_fn(uint64_t, 64);

vif_xy_fn(uint8_t, 8);
vif_xy_fn(uint16_t, 16);
vif_xy_fn(uint64_t, 64);

int compute_vif2(const int vif_filter[4][17], const void *ref, const void *main,
                 int w, int h, ptrdiff_t ref_stride, ptrdiff_t main_stride,
                 double *score, double *score_num, double *score_den,
                 double *scores, uint64_t *data_buf, uint64_t *temp, uint8_t type)
{
    char *data_top;

    uint64_t *ref_scale;
    uint64_t *main_scale;
    uint64_t *ref_sq;
    uint64_t *main_sq;
    uint64_t *ref_main;

    uint64_t *mu1;
    uint64_t *mu2;
    uint64_t *mu1_sq;
    uint64_t *mu2_sq;
    uint64_t *mu1_mu2;
    uint64_t *ref_sq_filt;
    uint64_t *main_sq_filt;
    uint64_t *ref_main_filt;
    uint64_t *num_array;
    uint64_t *den_array;
    uint64_t *temp_buf;

    const void *curr_ref_scale = ref;
    const void *curr_main_scale = main;
    int curr_ref_stride = ref_stride;
    int curr_main_stride = main_stride;

    ptrdiff_t buf_stride = ALIGN_CEIL(w * sizeof(uint64_t));
    size_t buf_sz = (size_t)buf_stride * h;

    double num = 0;
    double den = 0;

    int scale;
    int ret = 1;

    data_top = (char *) data_buf;

    ref_scale = (uint64_t *) data_top;
    data_top += buf_sz;

    main_scale = (uint64_t *) data_top;
    data_top += buf_sz;

    ref_sq = (uint64_t *) data_top;
    data_top += buf_sz;

    main_sq = (uint64_t *) data_top;
    data_top += buf_sz;

    ref_main = (uint64_t *) data_top;
    data_top += buf_sz;

    mu1 = (uint64_t *) data_top;
    data_top += buf_sz;

    mu2 = (uint64_t *) data_top;
    data_top += buf_sz;

    mu1_sq = (uint64_t *) data_top;
    data_top += buf_sz;

    mu2_sq = (uint64_t *) data_top;
    data_top += buf_sz;

    mu1_mu2 = (uint64_t *) data_top;
    data_top += buf_sz;

    ref_sq_filt = (uint64_t *) data_top;
    data_top += buf_sz;

    main_sq_filt = (uint64_t *) data_top;
    data_top += buf_sz;

    ref_main_filt = (uint64_t *) data_top;
    data_top += buf_sz;

    num_array = (uint64_t *) data_top;
    data_top += buf_sz;

    den_array = (uint64_t *) data_top;
    data_top += buf_sz;

    temp_buf = (uint64_t *) data_top;
    data_top += buf_sz;

    for (scale = 0; scale < 4; scale++) {
        const int *filter = vif_filter[scale];
        int filter_width = vif_filter_width[scale];

        int buf_valid_w = w;
        int buf_valid_h = h;

        if(!scale || scale == 1) {
            if(type <= 8) {
                vif_filter1d_8bit(filter, (const uint8_t *) curr_ref_scale, mu1, temp_buf, w, h, curr_ref_stride,
                                  buf_stride, filter_width, temp);
                vif_filter1d_8bit(filter, (const uint8_t *) curr_main_scale, mu2, temp_buf, w, h, curr_main_stride,
                                  buf_stride, filter_width, temp);
                vif_xx_yy_xy_8bit((const uint8_t *) curr_ref_scale, (const uint8_t *) curr_main_scale, ref_sq, main_sq, ref_main,
                                  w, h, curr_ref_stride, curr_main_stride, buf_stride,
                                  buf_stride, buf_stride);
            } else {
                vif_filter1d_16bit(filter, (const uint16_t *) curr_ref_scale, mu1, temp_buf, w, h, curr_ref_stride,
                                   buf_stride, filter_width, temp);
                vif_filter1d_16bit(filter, (const uint16_t *) curr_main_scale, mu2, temp_buf, w, h, curr_main_stride,
                                   buf_stride, filter_width, temp);
                vif_xx_yy_xy_16bit((const uint16_t *) curr_ref_scale, (const uint16_t *) curr_main_scale, ref_sq, main_sq, ref_main,
                                   w, h, curr_ref_stride, curr_main_stride, buf_stride,
                                   buf_stride, buf_stride);
            }
        } else if(scale > 1) {
            vif_filter1d_64bit(filter, curr_ref_scale, mu1, temp_buf, w, h,
                               curr_ref_stride, buf_stride, filter_width, temp);
            vif_filter1d_64bit(filter, curr_main_scale, mu2, temp_buf, w, h,
                               curr_main_stride, buf_stride, filter_width, temp);
        }
        if(scale >= 1) {
            vif_dec2(mu1, ref_scale, buf_valid_w, buf_valid_h, buf_stride,
                     buf_stride);
            vif_dec2(mu2, main_scale, buf_valid_w, buf_valid_h, buf_stride,
                     buf_stride);

            w  = buf_valid_w / 2;
            h  = buf_valid_h / 2;

            buf_valid_w = w;
            buf_valid_h = h;

            curr_ref_scale = ref_scale;
            curr_main_scale = main_scale;

            curr_ref_stride = buf_stride;
            curr_main_stride = buf_stride;

            vif_filter1d_64bit(filter, curr_ref_scale, mu1, temp_buf, w, h,
                               curr_ref_stride, buf_stride, filter_width, temp);
            vif_filter1d_64bit(filter, curr_main_scale, mu2, temp_buf, w, h,
                               curr_main_stride, buf_stride, filter_width, temp);
            vif_xx_yy_xy_64bit(curr_ref_scale, curr_main_scale, ref_sq, main_sq,
                               ref_main, w, h, curr_ref_stride, curr_main_stride,
                               buf_stride, buf_stride, buf_stride);
        }

        vif_xx_yy_xy_64bit(mu1, mu2, mu1_sq, mu2_sq, mu1_mu2, w, h, buf_stride,
                           buf_stride, buf_stride, buf_stride, buf_stride);

        vif_filter1d_64bit(filter, ref_sq, ref_sq_filt, temp_buf, w, h,
                           buf_stride, buf_stride, filter_width, temp);
        vif_filter1d_64bit(filter, main_sq, main_sq_filt, temp_buf, w, h,
                           buf_stride, buf_stride, filter_width, temp);
        vif_filter1d_64bit(filter, ref_main, ref_main_filt, temp_buf, w, h,
                           buf_stride, buf_stride, filter_width, temp);

        vif_statistic(mu1_sq, mu2_sq, mu1_mu2, ref_sq_filt, main_sq_filt,
                      ref_main_filt, num_array, den_array, w, h, buf_stride,
                      buf_stride, buf_stride, buf_stride, buf_stride,
                      buf_stride, buf_stride, buf_stride);

        num = vif_sum(num_array, buf_valid_w, buf_valid_h, buf_stride);
        den = vif_sum(den_array, buf_valid_w, buf_valid_h, buf_stride);

        scores[2 * scale] = num;
        scores[2 * scale + 1] = den;
    }

    *score_num = 0.0;
    *score_den = 0.0;
    for (scale = 0; scale < 4; scale++) {
        *score_num += scores[2 * scale];
        *score_den += scores[2 * scale + 1];
    }

    if (*score_den == 0.0) {
        *score = 1.0;
    } else {
        *score = (*score_num) / (*score_den);
    }

    ret = 0;

    return ret;
}

static void set_meta(AVDictionary **metadata, const char *key, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%0.2f", d);
    av_dict_set(metadata, key, value, 0);
}

static AVFrame *do_vif(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    VIFContext *s = ctx->priv;
    AVDictionary **metadata = &main->metadata;

    double score = 0.0;
    double score_num = 0.0;
    double score_den = 0.0;
    double scores[8];

    int w = s->width;
    int h = s->height;

    ptrdiff_t ref_stride = ref->linesize[0];
    ptrdiff_t main_stride = main->linesize[0];

    compute_vif2(s->vif_filter, ref->data[0], main->data[0], w, h, ref_stride,
                 main_stride, &score, &score_num, &score_den, scores,
                 s->data_buf, s->temp, s->desc->comp[0].depth);
    set_meta(metadata, "lavfi.vif.score", score);

    s->nb_frames++;

    s->vif_sum += score;

    return main;
}

static av_cold int init(AVFilterContext *ctx)
{
    VIFContext *s = ctx->priv;

    int i,j;
    for(i = 0; i < 4; i++) {
        for(j = 0; j < vif_filter_width[i]; j++){
            s->vif_filter[i][j] = lrint(vif_filter_table[i][j] * (1 << N));
        }
    }

    s->dinput.process = do_vif;

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
    VIFContext *s = ctx->priv;
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

    stride = ALIGN_CEIL(s->width * sizeof(uint64_t));
    data_sz = (size_t)stride * s->height;

    if (SIZE_MAX / data_sz < 15) {
        av_log(ctx, AV_LOG_ERROR, "error: SIZE_MAX / buf_sz < 15\n");
        return AVERROR(EINVAL);
    }

    if (!(s->data_buf = av_malloc(data_sz * 16))) {
        return AVERROR(ENOMEM);
    }

    if (!(s->temp = av_malloc(s->width * sizeof(uint64_t)))) {
        return AVERROR(ENOMEM);
    }

    return 0;
}


static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    VIFContext *s = ctx->priv;
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
    VIFContext *s = inlink->dst->priv;
    return ff_dualinput_filter_frame(&s->dinput, inlink, inpicref);
}

static int request_frame(AVFilterLink *outlink)
{
    VIFContext *s = outlink->src->priv;
    return ff_dualinput_request_frame(&s->dinput, outlink);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    VIFContext *s = ctx->priv;

    if (s->nb_frames > 0) {
        av_log(ctx, AV_LOG_INFO, "VIF AVG: %.3f\n", s->vif_sum / s->nb_frames);
    }

    av_free(s->data_buf);
    av_free(s->temp);

    ff_dualinput_uninit(&s->dinput);
}

static const AVFilterPad vif_inputs[] = {
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

static const AVFilterPad vif_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
        .request_frame = request_frame,
    },
    { NULL }
};

AVFilter ff_vf_vif = {
    .name          = "vif",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the VIF between two video streams."),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(VIFContext),
    .priv_class    = &vif_class,
    .inputs        = vif_inputs,
    .outputs       = vif_outputs,
};
