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
 * Calculate the ADM between two input videos.
 */

#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "dualinput.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "adm.h"
#include "video.h"
#include <emmintrin.h>

typedef struct ADMContext {
    const AVClass *class;
    FFDualInputContext dinput;
    const AVPixFmtDescriptor *desc;
    int width;
    int height;
    float *ref_data;
    float *main_data;
    float *data_buf;
    float *temp_lo;
    float *temp_hi;
    double adm_sum;
    uint64_t nb_frames;
} ADMContext;

static const AVOption adm_options[] = {
    { NULL }
};

AVFILTER_DEFINE_CLASS(adm);

#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define OPT_RANGE_PIXEL_OFFSET (-128)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static float rcp(float x)
{
    float xi = _mm_cvtss_f32(_mm_rcp_ss(_mm_load_ss(&x)));
    return xi + xi * (1.0f - x * xi);
}

#define DIVS(n, d) ((n) * rcp(d))

av_always_inline static float dwt_quant_step(const struct dwt_model_params *params,
                                             int lambda, int theta)
{
    float r = VIEW_DIST * REF_DISPLAY_HEIGHT * M_PI / 180.0;

    float temp = log10(pow(2.0, lambda+1) * params->f0 * params->g[theta] / r);
    float Q = 2.0 * params->a * pow(10.0, params->k * temp * temp) /
        dwt_7_9_basis_function_amplitudes[lambda][theta];

    return Q;
}

static float get_cube(float val)
{
    return val * val * val;
}

static float adm_sum_cube(const float *x, int w, int h, int stride,
                          double border_factor)
{
    int px_stride = stride / sizeof(float);
    int left   = w * border_factor - 0.5;
    int top    = h * border_factor - 0.5;
    int right  = w - left;
    int bottom = h - top;

    int i, j;

    float val;
    float accum = 0;

    for (i = top; i < bottom; i++) {
        for (j = left; j < right; j++) {
            val = fabsf(x[i * px_stride + j]);
            accum += get_cube(val);
        }
    }

    return powf(accum, 1.0f / 3.0f) + powf((bottom - top) * (right - left) /
                                           32.0f, 1.0f / 3.0f);
}

static void adm_decouple(const adm_dwt_band_t *ref, const adm_dwt_band_t *dis,
                         const adm_dwt_band_t *r, const adm_dwt_band_t *a,
                         int w, int h, int ref_stride, int dis_stride,
                         int r_stride, int a_stride)
{
    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
    const float eps = 1e-30;

    int ref_px_stride = ref_stride / sizeof(float);
    int dis_px_stride = dis_stride / sizeof(float);
    int r_px_stride = r_stride / sizeof(float);
    int a_px_stride = a_stride / sizeof(float);

    float oh, ov, od, th, tv, td;
    float kh, kv, kd, tmph, tmpv, tmpd;
    float ot_dp, o_mag_sq, t_mag_sq;
    int angle_flag;
    int i, j;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            oh = ref->band_h[i * ref_px_stride + j];
            ov = ref->band_v[i * ref_px_stride + j];
            od = ref->band_d[i * ref_px_stride + j];
            th = dis->band_h[i * dis_px_stride + j];
            tv = dis->band_v[i * dis_px_stride + j];
            td = dis->band_d[i * dis_px_stride + j];

            kh = DIVS(th, oh + eps);
            kv = DIVS(tv, ov + eps);
            kd = DIVS(td, od + eps);

            kh = kh < 0.0f ? 0.0f : (kh > 1.0f ? 1.0f : kh);


            kv = kv < 0.0f ? 0.0f : (kv > 1.0f ? 1.0f : kv);
            kd = kd < 0.0f ? 0.0f : (kd > 1.0f ? 1.0f : kd);

            tmph = kh * oh;
            tmpv = kv * ov;
            tmpd = kd * od;

            ot_dp = oh * th + ov * tv;
            o_mag_sq = oh * oh + ov * ov;
            t_mag_sq = th * th + tv * tv;

            angle_flag = (ot_dp >= 0.0f) && (ot_dp * ot_dp >= cos_1deg_sq *
                                             o_mag_sq * t_mag_sq);

            if (angle_flag) {
                tmph = th;
                tmpv = tv;
                tmpd = td;
            }

            r->band_h[i * r_px_stride + j] = tmph;
            r->band_v[i * r_px_stride + j] = tmpv;
            r->band_d[i * r_px_stride + j] = tmpd;

            a->band_h[i * a_px_stride + j] = th - tmph;
            a->band_v[i * a_px_stride + j] = tv - tmpv;
            a->band_d[i * a_px_stride + j] = td - tmpd;
        }
    }
}

static void adm_csf(const adm_dwt_band_t *src, const adm_dwt_band_t *dst,
                    int orig_h, int scale, int w, int h, int src_stride,
                    int dst_stride)
{
    const float *src_angles[3] = { src->band_h, src->band_v, src->band_d };
    float *dst_angles[3]       = { dst->band_h, dst->band_v, dst->band_d };

    const float *src_ptr;
    float *dst_ptr;

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float factor1 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1);
    float factor2 = dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2);
    float rfactor[3] = {1.0f / factor1, 1.0f / factor1, 1.0f / factor2};

    int i, j, theta;

    for (theta = 0; theta < 3; theta++) {
        src_ptr = src_angles[theta];
        dst_ptr = dst_angles[theta];

        for (i = 0; i < h; i++) {
            for (j = 0; j < w; j++) {
                dst_ptr[i * dst_px_stride + j] = rfactor[theta] *
                    src_ptr[i * src_px_stride + j];
            }
        }
    }
}

static void adm_cm_thresh(const adm_dwt_band_t *src, float *dst, int w, int h,
                          int src_stride, int dst_stride)
{
    const float *angles[3] = { src->band_h, src->band_v, src->band_d };
    const float *src_ptr;

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float filt_coeff, imgcoeff;

    int theta, i, j, filt_i, filt_j, src_i, src_j;

    for (i = 0; i < h; i++) {

        for (j = 0; j < w; j++) {
            dst[i * dst_px_stride + j] = 0;
        }

        for (theta = 0; theta < 3; ++theta) {
            src_ptr = angles[theta];

            for (j = 0; j < w; j++) {
                float accum = 0;

                for (filt_i = 0; filt_i < 3; filt_i++) {
                    for (filt_j = 0; filt_j < 3; filt_j++) {
                        filt_coeff = (filt_i == 1 && filt_j == 1) ? 1.0f / 15.0f : 1.0f /
                            30.0f;

                        src_i = i - 1 + filt_i;
                        src_j = j - 1 + filt_j;

                        src_i = FFABS(src_i);
                        if (src_i >= h) {
                            src_i = 2 * h - src_i - 1;
                        }
                        src_j = FFABS(src_j);
                        if (src_j >= w) {
                            src_j = 2 * w - src_j - 1;
                        }
                        imgcoeff = fabsf(src_ptr[src_i * src_px_stride + src_j]);

                        accum += filt_coeff * imgcoeff;
                    }
                }

                dst[i * dst_px_stride + j] += accum;
            }
        }
    }
}

static void adm_cm(const adm_dwt_band_t *src, const adm_dwt_band_t *dst,
                   const float *thresh, int w, int h, int src_stride,
                   int dst_stride, int thresh_stride)
{
    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);
    int thresh_px_stride = thresh_stride / sizeof(float);

    float xh, xv, xd, thr;

    int i, j;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            xh  = src->band_h[i * src_px_stride + j];
            xv  = src->band_v[i * src_px_stride + j];
            xd  = src->band_d[i * src_px_stride + j];
            thr = thresh[i * thresh_px_stride + j];

            xh = fabsf(xh) - thr;
            xv = fabsf(xv) - thr;
            xd = fabsf(xd) - thr;

            xh = xh < 0.0f ? 0.0f : xh;
            xv = xv < 0.0f ? 0.0f : xv;
            xd = xd < 0.0f ? 0.0f : xd;

            dst->band_h[i * dst_px_stride + j] = xh;
            dst->band_v[i * dst_px_stride + j] = xv;
            dst->band_d[i * dst_px_stride + j] = xd;
        }
    }
}

static void adm_dwt2(const float *src, const adm_dwt_band_t *dst, int w, int h,
                     int src_stride, int dst_stride, ADMContext *s)
{
    const float *filter_lo = dwt2_db2_coeffs_lo;
    const float *filter_hi = dwt2_db2_coeffs_hi;
    int filt_width = sizeof(dwt2_db2_coeffs_lo) / sizeof(float);

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float filt_coeff_lo, filt_coeff_hi, imgcoeff;

    int i, j, filt_i, filt_j, src_i, src_j;

    for (i = 0; i < (h + 1) / 2; i++) {
        /** Vertical pass. */
        for (j = 0; j < w; j++) {
            float accum_lo = 0;
            float accum_hi = 0;

            for (filt_i = 0; filt_i < filt_width; filt_i++) {
                filt_coeff_lo = filter_lo[filt_i];
                filt_coeff_hi = filter_hi[filt_i];

                src_i = 2 * i - 1 + filt_i;

                src_i = FFABS(src_i);
                if (src_i >= h) {
                    src_i = 2 * h - src_i - 1;
                }

                imgcoeff = src[src_i * src_px_stride + j];

                accum_lo += filt_coeff_lo * imgcoeff;
                accum_hi += filt_coeff_hi * imgcoeff;
            }

            s->temp_lo[j] = accum_lo;
            s->temp_hi[j] = accum_hi;
        }

        /** Horizontal pass (lo). */
        for (j = 0; j < (w + 1) / 2; j++) {
            float accum_lo = 0;
            float accum_hi = 0;

            for (filt_j = 0; filt_j < filt_width; filt_j++) {
                filt_coeff_lo = filter_lo[filt_j];
                filt_coeff_hi = filter_hi[filt_j];

                src_j = 2 * j - 1 + filt_j;

                src_j = FFABS(src_j);
                if (src_j >= w) {
                    src_j = 2 * w - src_j - 1;
                }

                imgcoeff = s->temp_lo[src_j];

                accum_lo += filt_coeff_lo * imgcoeff;
                accum_hi += filt_coeff_hi * imgcoeff;
            }

            dst->band_a[i * dst_px_stride + j] = accum_lo;
            dst->band_v[i * dst_px_stride + j] = accum_hi;
        }

        /** Horizontal pass (hi). */
        for (j = 0; j < (w + 1) / 2; j++) {
            float accum_lo = 0;
            float accum_hi = 0;

            for (filt_j = 0; filt_j < filt_width; filt_j++) {
                filt_coeff_lo = filter_lo[filt_j];
                filt_coeff_hi = filter_hi[filt_j];

                src_j = 2 * j - 1 + filt_j;

                src_j = FFABS(src_j);
                if (src_j >= w) {
                    src_j = 2 * w - src_j - 1;
                }

                imgcoeff = s->temp_hi[src_j];

                accum_lo += filt_coeff_lo * imgcoeff;
                accum_hi += filt_coeff_hi * imgcoeff;
            }

            dst->band_h[i * dst_px_stride + j] = accum_lo;
            dst->band_d[i * dst_px_stride + j] = accum_hi;
        }
    }

}

static void adm_buffer_copy(const void *src, void *dst, int linewidth, int h,
                            int src_stride, int dst_stride)
{
    const char *src_p = src;
    char *dst_p = dst;
    int i;

    for (i = 0; i < h; i++) {
        memcpy(dst_p, src_p, linewidth);
        src_p += src_stride;
        dst_p += dst_stride;
    }
}

static char *init_dwt_band(adm_dwt_band_t *band, char *data_top, size_t buf_sz)
{
    band->band_a = (float *) data_top;
    data_top += buf_sz;
    band->band_h = (float *) data_top;
    data_top += buf_sz;
    band->band_v = (float *) data_top;
    data_top += buf_sz;
    band->band_d = (float *) data_top;
    data_top += buf_sz;
    return data_top;
}

int compute_adm(const float *ref, const float *dis, int w, int h,
                int ref_stride, int dis_stride, double *score,
                double *score_num, double *score_den, double *scores,
                double border_factor, void *ctx)
{
    ADMContext *s = (ADMContext *) ctx;
    double numden_limit = 1e-2 * (w * h) / (1920.0 * 1080.0);

    char *data_top;

    float *ref_scale;
    float *dis_scale;

    adm_dwt_band_t ref_dwt2;
    adm_dwt_band_t dis_dwt2;

    adm_dwt_band_t decouple_r;
    adm_dwt_band_t decouple_a;

    adm_dwt_band_t csf_o;
    adm_dwt_band_t csf_r;
    adm_dwt_band_t csf_a;

    float *mta;

    adm_dwt_band_t cm_r;

    const float *curr_ref_scale = (float *) ref;
    const float *curr_dis_scale = (float *) dis;
    int curr_ref_stride = ref_stride;
    int curr_dis_stride = dis_stride;

    int orig_h = h;

    int buf_stride = ALIGN_CEIL(((w + 1) / 2) * sizeof(float));
    size_t buf_sz = (size_t)buf_stride * ((h + 1) / 2);

    double num = 0;
    double den = 0;

    int scale;
    int ret = 1;

    data_top = (char *) (s->data_buf);

    ref_scale = (float *) data_top;
    data_top += buf_sz;
    dis_scale = (float *) data_top;
    data_top += buf_sz;

    data_top = init_dwt_band(&ref_dwt2, data_top, buf_sz);
    data_top = init_dwt_band(&dis_dwt2, data_top, buf_sz);
    data_top = init_dwt_band(&decouple_r, data_top, buf_sz);
    data_top = init_dwt_band(&decouple_a, data_top, buf_sz);
    data_top = init_dwt_band(&csf_o, data_top, buf_sz);
    data_top = init_dwt_band(&csf_r, data_top, buf_sz);
    data_top = init_dwt_band(&csf_a, data_top, buf_sz);

    mta = (float *) data_top;
    data_top += buf_sz;

    data_top = init_dwt_band(&cm_r, data_top, buf_sz);

    for (scale = 0; scale < 4; scale++) {
        float num_scale = 0.0;
        float den_scale = 0.0;

        adm_dwt2(curr_ref_scale, &ref_dwt2, w, h, curr_ref_stride, buf_stride, s);
        adm_dwt2(curr_dis_scale, &dis_dwt2, w, h, curr_dis_stride, buf_stride, s);

        w = (w + 1) / 2;
        h = (h + 1) / 2;

        adm_decouple(&ref_dwt2, &dis_dwt2, &decouple_r, &decouple_a, w, h,
                     buf_stride, buf_stride, buf_stride, buf_stride);

        adm_csf(&ref_dwt2, &csf_o, orig_h, scale, w, h, buf_stride, buf_stride);
        adm_csf(&decouple_r, &csf_r, orig_h, scale, w, h, buf_stride, buf_stride);
        adm_csf(&decouple_a, &csf_a, orig_h, scale, w, h, buf_stride, buf_stride);

        adm_cm_thresh(&csf_a, mta, w, h, buf_stride, buf_stride);
        adm_cm(&csf_r, &cm_r, mta, w, h, buf_stride, buf_stride, buf_stride);

        num_scale += adm_sum_cube(cm_r.band_h, w, h, buf_stride, border_factor);
        num_scale += adm_sum_cube(cm_r.band_v, w, h, buf_stride, border_factor);
        num_scale += adm_sum_cube(cm_r.band_d, w, h, buf_stride, border_factor);

        den_scale += adm_sum_cube(csf_o.band_h, w, h, buf_stride, border_factor);
        den_scale += adm_sum_cube(csf_o.band_v, w, h, buf_stride, border_factor);
        den_scale += adm_sum_cube(csf_o.band_d, w, h, buf_stride, border_factor);

        num += num_scale;
        den += den_scale;

        adm_buffer_copy(ref_dwt2.band_a, ref_scale, w * sizeof(float), h,
                        buf_stride, buf_stride);
        adm_buffer_copy(dis_dwt2.band_a, dis_scale, w * sizeof(float), h,
                        buf_stride, buf_stride);

        curr_ref_scale = ref_scale;
        curr_dis_scale = dis_scale;
        curr_ref_stride = buf_stride;
        curr_dis_stride = buf_stride;

        scores[2*scale+0] = num_scale;
        scores[2*scale+1] = den_scale;
    }

    num = num < numden_limit ? 0 : num;
    den = den < numden_limit ? 0 : den;

    if (den == 0.0) {
        *score = 1.0f;
    } else {
        *score = num / den;
    }
    *score_num = num;
    *score_den = den;

    ret = 0;

    return ret;
}

#define offset_fn(type, bits) \
    static void offset_##bits##bit(ADMContext *s, const AVFrame *ref, AVFrame *main, int stride) \
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

static AVFrame *do_adm(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    ADMContext *s = ctx->priv;
    AVDictionary **metadata = &main->metadata;

    double score = 0.0;
    double score_num = 0;
    double score_den = 0;
    double scores[2*4];

    int w = s->width;
    int h = s->height;

    int stride;

    stride = ALIGN_CEIL(w * sizeof(float));

    /** Offset ref and main pixel by OPT_RANGE_PIXEL_OFFSET */
    if (s->desc->comp[0].depth <= 8) {
        offset_8bit(s, ref, main, stride);
    } else {
        offset_10bit(s, ref, main, stride);
    }

    compute_adm(s->ref_data, s->main_data, w, h, stride, stride, &score,
                &score_num, &score_den, scores, ADM_BORDER_FACTOR, s);

    set_meta(metadata, "lavfi.adm.score", 0, score);

    s->nb_frames++;

    s->adm_sum += score;

    return main;
}

static av_cold int init(AVFilterContext *ctx)
{
    ADMContext *s = ctx->priv;

    s->dinput.process = do_adm;

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
    ADMContext *s = ctx->priv;
    int buf_stride;
    size_t buf_sz;
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
        av_log(ctx, AV_LOG_ERROR, "ref data allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    if (!(s->main_data = av_malloc(data_sz))) {
        av_log(ctx, AV_LOG_ERROR, "main data allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    buf_stride = ALIGN_CEIL(((s->width + 1) / 2) * sizeof(float));
    buf_sz = (size_t)buf_stride * ((s->height + 1) / 2);

    if (SIZE_MAX / buf_sz < 35) {
        av_log(ctx, AV_LOG_ERROR, "error: SIZE_MAX / buf_sz_one < 35, buf_sz_one = %lu.\n", buf_sz);
        return AVERROR(EINVAL);
    }

    if (!(s->data_buf = av_malloc(buf_sz * 35))) {
        av_log(ctx, AV_LOG_ERROR, "data_buf allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    if (!(s->temp_lo = av_malloc(stride))) {
        av_log(ctx, AV_LOG_ERROR, "temp lo allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    if (!(s->temp_hi = av_malloc(stride))) {
        av_log(ctx, AV_LOG_ERROR, "temp hi allocation failed.\n");
        return AVERROR(ENOMEM);
    }

    return 0;
}


static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    ADMContext *s = ctx->priv;
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
    ADMContext *s = inlink->dst->priv;
    return ff_dualinput_filter_frame(&s->dinput, inlink, inpicref);
}

static int request_frame(AVFilterLink *outlink)
{
    ADMContext *s = outlink->src->priv;
    return ff_dualinput_request_frame(&s->dinput, outlink);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    ADMContext *s = ctx->priv;

    if (s->nb_frames > 0) {
        av_log(ctx, AV_LOG_INFO, "ADM AVG: %.3f\n", s->adm_sum / s->nb_frames);
    }

    av_free(s->ref_data);
    av_free(s->main_data);
    av_free(s->data_buf);
    av_free(s->temp_lo);
    av_free(s->temp_hi);

    ff_dualinput_uninit(&s->dinput);
}

static const AVFilterPad adm_inputs[] = {
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

static const AVFilterPad adm_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
        .request_frame = request_frame,
    },
    { NULL }
};

AVFilter ff_vf_adm = {
    .name          = "adm",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the ADM between two video streams."),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(ADMContext),
    .priv_class    = &adm_class,
    .inputs        = adm_inputs,
    .outputs       = adm_outputs,
};
