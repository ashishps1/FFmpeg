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

#include <inttypes.h>
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
    int width;
    int height;
    char *format;
    float *data_buf;
    double adm_sum;
    uint64_t nb_frames;
} ADMContext;

typedef struct adm_dwt_band_t {
    float *band_a; /* Low-pass V + low-pass H. */
    float *band_v; /* Low-pass V + high-pass H. */
    float *band_h; /* High-pass V + low-pass H. */
    float *band_d; /* High-pass V + high-pass H. */
} adm_dwt_band_t;

#define OFFSET(x) offsetof(ADMContext, x)
#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define OPT_RANGE_PIXEL_OFFSET (-128)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM
#define M_PI 3.1415926535897932384626433832795028841971693993751
/* Percentage of frame to discard on all 4 sides */
#define ADM_BORDER_FACTOR (0.1)

static const float dwt2_db2_coeffs_lo[4] = { 0.482962913144690, 0.836516303737469, 0.224143868041857, -0.129409522550921 };
static const float dwt2_db2_coeffs_hi[4] = { -0.129409522550921, -0.224143868041857, 0.836516303737469, -0.482962913144690 };

static float rcp(float x)
{
    float xi = _mm_cvtss_f32(_mm_rcp_ss(_mm_load_ss(&x)));
    return xi + xi * (1.0f - x * xi);
}

#define DIVS(n, d) ((n) * rcp(d))

#define VIEW_DIST 3.0f

#define REF_DISPLAY_HEIGHT 1080

struct dwt_model_params {
    float a;
    float k;
    float f0;
    float g[4];
};

static const struct dwt_model_params dwt_7_9_YCbCr_threshold[3] = {
    { .a = 0.495, .k = 0.466, .f0 = 0.401, .g = { 1.501, 1.0, 0.534, 1.0} },
    { .a = 1.633, .k = 0.353, .f0 = 0.209, .g = { 1.520, 1.0, 0.502, 1.0} },
    { .a = 0.944, .k = 0.521, .f0 = 0.404, .g = { 1.868, 1.0, 0.516, 1.0} }
};

static const float dwt_7_9_basis_function_amplitudes[6][4] = {
    { 0.62171,  0.67234,  0.72709,  0.67234  },
    { 0.34537,  0.41317,  0.49428,  0.41317  },
    { 0.18004,  0.22727,  0.28688,  0.22727  },
    { 0.091401, 0.11792,  0.15214,  0.11792  },
    { 0.045943, 0.059758, 0.077727, 0.059758 },
    { 0.023013, 0.030018, 0.039156, 0.030018 }
};

static inline double get_adm_avg(double ansnr_sum, uint64_t nb_frames)
{
    return ansnr_sum / nb_frames;
}

static inline float dwt_quant_step(const struct dwt_model_params *params, int lambda, int theta)
{
    float r = VIEW_DIST * REF_DISPLAY_HEIGHT * M_PI / 180.0;

    float temp = log10(pow(2.0,lambda+1)*params->f0*params->g[theta]/r);
    float Q = 2.0*params->a*pow(10.0,params->k*temp*temp)/dwt_7_9_basis_function_amplitudes[lambda][theta];

    return Q;
}

static const AVOption adm_options[] = {
    { NULL }
};

AVFILTER_DEFINE_CLASS(adm);

static float adm_sum_cube(const float *x, int w, int h, int stride, double border_factor)
{
    int px_stride = stride / sizeof(float);
    int left   = w * border_factor - 0.5;
    int top    = h * border_factor - 0.5;
    int right  = w - left;
    int bottom = h - top;

    int i, j;

    float val;
    float accum = 0;

    for (i = top; i < bottom; ++i) {
        float accum_inner = 0;

        for (j = left; j < right; ++j) {
            val = fabsf(x[i * px_stride + j]);

            accum_inner += val * val * val;
        }

        accum += accum_inner;
    }

    return powf(accum, 1.0f / 3.0f) + powf((bottom - top) * (right - left) / 32.0f, 1.0f / 3.0f);
}

static void adm_decouple(const adm_dwt_band_t *ref, const adm_dwt_band_t *dis, const adm_dwt_band_t *r, const adm_dwt_band_t *a, int w, int h, int ref_stride, int dis_stride, int r_stride, int a_stride)
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

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
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

            angle_flag = (ot_dp >= 0.0f) && (ot_dp * ot_dp >= cos_1deg_sq * o_mag_sq * t_mag_sq);

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

static void adm_csf(const adm_dwt_band_t *src, const adm_dwt_band_t *dst, int orig_h, int scale, int w, int h, int src_stride, int dst_stride)
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

    for (theta = 0; theta < 3; ++theta) {
        src_ptr = src_angles[theta];
        dst_ptr = dst_angles[theta];

        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                dst_ptr[i * dst_px_stride + j] = rfactor[theta] * src_ptr[i * src_px_stride + j];
            }
        }
    }
}

static void adm_cm_thresh(const adm_dwt_band_t *src, float *dst, int w, int h, int src_stride, int dst_stride)
{
    const float *angles[3] = { src->band_h, src->band_v, src->band_d };
    const float *src_ptr;

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float fcoeff, imgcoeff;

    int theta, i, j, fi, fj, ii, jj;

    for (i = 0; i < h; ++i) {
        /* Zero output row. */
        for (j = 0; j < w; ++j) {
            dst[i * dst_px_stride + j] = 0;
        }

        for (theta = 0; theta < 3; ++theta) {
            src_ptr = angles[theta];

            for (j = 0; j < w; ++j) {
                float accum = 0;

                for (fi = 0; fi < 3; ++fi) {
                    for (fj = 0; fj < 3; ++fj) {
                        fcoeff = (fi == 1 && fj == 1) ? 1.0f / 15.0f : 1.0f / 30.0f;

                        ii = i - 1 + fi;
                        jj = j - 1 + fj;

                        if (ii < 0)
                            ii = -ii;
                        else if (ii >= h)
                            ii = 2 * h - ii - 1;
                        if (jj < 0)
                            jj = -jj;
                        else if (jj >= w)
                            jj = 2 * w - jj - 1;
                        imgcoeff = fabsf(src_ptr[ii * src_px_stride + jj]);

                        accum += fcoeff * imgcoeff;
                    }
                }

                dst[i * dst_px_stride + j] += accum;
            }
        }
    }
}

static void adm_cm(const adm_dwt_band_t *src, const adm_dwt_band_t *dst, const float *thresh, int w, int h, int src_stride, int dst_stride, int thresh_stride)
{
    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);
    int thresh_px_stride = thresh_stride / sizeof(float);

    float xh, xv, xd, thr;

    int i, j;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
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

static void adm_dwt2(const float *src, const adm_dwt_band_t *dst, int w, int h, int src_stride, int dst_stride)
{
    const float *filter_lo = dwt2_db2_coeffs_lo;
    const float *filter_hi = dwt2_db2_coeffs_hi;
    int fwidth = sizeof(dwt2_db2_coeffs_lo) / sizeof(float);

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float *tmplo = av_malloc(ALIGN_CEIL(sizeof(float) * w));
    float *tmphi = av_malloc(ALIGN_CEIL(sizeof(float) * w));
    float fcoeff_lo, fcoeff_hi, imgcoeff;

    int i, j, fi, fj, ii, jj;

    for (i = 0; i < (h + 1) / 2; i++) {
        /* Vertical pass. */
        for (j = 0; j < w; j++) {
            float accum_lo = 0;
            float accum_hi = 0;

            for (fi = 0; fi < fwidth; fi++) {
                fcoeff_lo = filter_lo[fi];
                fcoeff_hi = filter_hi[fi];

                ii = 2 * i - 1 + fi;

                if (ii < 0)
                    ii = -ii;
                else if (ii >= h)
                    ii = 2 * h - ii - 1;

                imgcoeff = src[ii * src_px_stride + j];

                accum_lo += fcoeff_lo * imgcoeff;
                accum_hi += fcoeff_hi * imgcoeff;
            }

            tmplo[j] = accum_lo;
            tmphi[j] = accum_hi;
        }

        /* Horizontal pass (lo). */
        for (j = 0; j < (w + 1) / 2; j++) {
            float accum_lo = 0;
            float accum_hi = 0;

            for (fj = 0; fj < fwidth; fj++) {
                fcoeff_lo = filter_lo[fj];
                fcoeff_hi = filter_hi[fj];

                jj = 2 * j - 1 + fj;

                if (jj < 0)
                    jj = -jj;
                else if (jj >= w)
                    jj = 2 * w - jj - 1;

                imgcoeff = tmplo[jj];

                accum_lo += fcoeff_lo * imgcoeff;
                accum_hi += fcoeff_hi * imgcoeff;
            }

            dst->band_a[i * dst_px_stride + j] = accum_lo;
            dst->band_v[i * dst_px_stride + j] = accum_hi;
        }

        /* Horizontal pass (hi). */
        for (j = 0; j < (w + 1) / 2; j++) {
            float accum_lo = 0;
            float accum_hi = 0;

            for (fj = 0; fj < fwidth; fj++) {
                fcoeff_lo = filter_lo[fj];
                fcoeff_hi = filter_hi[fj];

                jj = 2 * j - 1 + fj;

                if (jj < 0)
                    jj = -jj;
                else if (jj >= w)
                    jj = 2 * w - jj - 1;

                imgcoeff = tmphi[jj];

                accum_lo += fcoeff_lo * imgcoeff;
                accum_hi += fcoeff_hi * imgcoeff;
            }

            dst->band_h[i * dst_px_stride + j] = accum_lo;
            dst->band_d[i * dst_px_stride + j] = accum_hi;
        }
    }

    av_free(tmplo);
    av_free(tmphi);
}

static void adm_buffer_copy(const void *src, void *dst, int linewidth, int h, int src_stride, int dst_stride)
{
    const char *src_p = src;
    char *dst_p = dst;
    int i;

    for (i = 0; i < h; ++i) {
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

static int compute_adm(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride, double *score, double *score_num, double *score_den, double *scores, double border_factor, ADMContext *s)
{
    double numden_limit = 1e-2 * (w * h) / (1920.0 * 1080.0);
    float *data_buf = 0;
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

    const float *curr_ref_scale = (float *)ref;
    const float *curr_dis_scale = (float *)dis;
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

    mta = (float *)data_top;
    data_top += buf_sz;

    data_top = init_dwt_band(&cm_r, data_top, buf_sz);

    for (scale = 0; scale < 4; ++scale) {
        float num_scale = 0.0;
        float den_scale = 0.0;
        static int p =0;

        adm_dwt2(curr_ref_scale, &ref_dwt2, w, h, curr_ref_stride, buf_stride);
        adm_dwt2(curr_dis_scale, &dis_dwt2, w, h, curr_dis_stride, buf_stride);

        w = (w + 1) / 2;
        h = (h + 1) / 2;

        adm_decouple(&ref_dwt2, &dis_dwt2, &decouple_r, &decouple_a, w, h, buf_stride, buf_stride, buf_stride, buf_stride);

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

        /* Copy DWT2 approximation band to buffer for next scale. */
        adm_buffer_copy(ref_dwt2.band_a, ref_scale, w * sizeof(float), h, buf_stride, buf_stride);
        adm_buffer_copy(dis_dwt2.band_a, dis_scale, w * sizeof(float), h, buf_stride, buf_stride);

        curr_ref_scale = ref_scale;
        curr_dis_scale = dis_scale;
        curr_ref_stride = buf_stride;
        curr_dis_stride = buf_stride;

        scores[2*scale+0] = num_scale;
        scores[2*scale+1] = den_scale;
    }

    num = num < numden_limit ? 0 : num;
    den = den < numden_limit ? 0 : den;

    if (den == 0.0)
    {
        *score = 1.0f;
    }
    else
    {
        *score = num / den;
    }
    *score_num = num;
    *score_den = den;

    ret = 0;

    return ret;
}

static AVFrame *do_adm(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    ADMContext *s = ctx->priv;

    char *format = s->format;

    double score = 0.0;
    double score_num = 0;
    double score_den = 0;
    double scores[2*4];

    int w = s->width;
    int h = s->height;

    int stride;
    size_t data_sz;

    stride = ALIGN_CEIL(w * sizeof(float));
    data_sz = (size_t)stride * h;

    uint8_t *ref_ptr = ref->data[0];
    uint8_t *main_ptr = main->data[0];

    int ref_stride = ref->linesize[0];
    int main_stride = main->linesize[0];

    float *ref_data;
    float *main_data;

    ref_data = av_malloc(data_sz);
    main_data = av_malloc(data_sz);

    int i,j;

    float *ref_data_ptr = ref_data;
    float *main_data_ptr = main_data;

    for(i=0;i<h;i++){
        for(j=0;j<w;j++){
            ref_data_ptr[j] = (float)ref_ptr[j] + OPT_RANGE_PIXEL_OFFSET;
            main_data_ptr[j] = (float)main_ptr[j] + OPT_RANGE_PIXEL_OFFSET;
        }
        ref_data_ptr += stride / sizeof(float);
        main_data_ptr += stride / sizeof(float);
        ref_ptr += ref_stride / sizeof(uint8_t);
        main_ptr += main_stride / sizeof(uint8_t);
    }


    compute_adm(ref_data, main_data, w, h, stride, stride, &score,
                &score_num, &score_den, &scores, ADM_BORDER_FACTOR, s);

    s->nb_frames++;

    s->adm_sum += score;

    av_free(ref_data);
    av_free(main_data);

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
    s->format = av_get_pix_fmt_name(ctx->inputs[0]->format);

    buf_stride = ALIGN_CEIL(((s->width + 1) / 2) * sizeof(float));
    buf_sz = (size_t)buf_stride * ((s->height + 1) / 2);

    if (SIZE_MAX / buf_sz < 35) {
        av_log(ctx, AV_LOG_ERROR, "error: SIZE_MAX / buf_sz_one < 35, buf_sz_one = %lu.\n", buf_sz);
        return AVERROR(EINVAL);
    }

    if (!(s->data_buf = av_malloc(buf_sz * 35))) {
        av_log(ctx, AV_LOG_ERROR, "data_buf allocation failed.\n");
        return AVERROR(EINVAL);
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

    ff_dualinput_uninit(&s->dinput);

    av_free(s->data_buf);

    av_log(ctx, AV_LOG_INFO, "ADM AVG: %.3f\n", get_adm_avg(s->adm_sum, s->nb_frames));
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
