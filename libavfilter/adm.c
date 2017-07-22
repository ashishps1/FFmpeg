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

#include "libavutil/opt.h"
#include "adm.h"
#include <emmintrin.h>

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
                     int src_stride, int dst_stride, float *temp_lo, float* temp_hi)
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

            temp_lo[j] = accum_lo;
            temp_hi[j] = accum_hi;
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

                imgcoeff = temp_lo[src_j];

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

                imgcoeff = temp_hi[src_j];

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

int compute_adm1(const float *ref, const float *dis, int w, int h,
                 int ref_stride, int dis_stride, double *score,
                 double *score_num, double *score_den, double *scores,
                 float *data_buf, float *temp_lo, float* temp_hi)
{
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

    data_top = (char *) (data_buf);

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

        adm_dwt2(curr_ref_scale, &ref_dwt2, w, h, curr_ref_stride, buf_stride, temp_lo, temp_hi);
        adm_dwt2(curr_dis_scale, &dis_dwt2, w, h, curr_dis_stride, buf_stride, temp_lo, temp_hi);

        w = (w + 1) / 2;
        h = (h + 1) / 2;

        adm_decouple(&ref_dwt2, &dis_dwt2, &decouple_r, &decouple_a, w, h,
                     buf_stride, buf_stride, buf_stride, buf_stride);

        adm_csf(&ref_dwt2, &csf_o, orig_h, scale, w, h, buf_stride, buf_stride);
        adm_csf(&decouple_r, &csf_r, orig_h, scale, w, h, buf_stride, buf_stride);
        adm_csf(&decouple_a, &csf_a, orig_h, scale, w, h, buf_stride, buf_stride);

        adm_cm_thresh(&csf_a, mta, w, h, buf_stride, buf_stride);
        adm_cm(&csf_r, &cm_r, mta, w, h, buf_stride, buf_stride, buf_stride);

        num_scale += adm_sum_cube(cm_r.band_h, w, h, buf_stride, ADM_BORDER_FACTOR);
        num_scale += adm_sum_cube(cm_r.band_v, w, h, buf_stride, ADM_BORDER_FACTOR);
        num_scale += adm_sum_cube(cm_r.band_d, w, h, buf_stride, ADM_BORDER_FACTOR);

        den_scale += adm_sum_cube(csf_o.band_h, w, h, buf_stride, ADM_BORDER_FACTOR);
        den_scale += adm_sum_cube(csf_o.band_v, w, h, buf_stride, ADM_BORDER_FACTOR);
        den_scale += adm_sum_cube(csf_o.band_d, w, h, buf_stride, ADM_BORDER_FACTOR);

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
