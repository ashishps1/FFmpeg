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

#include "libavutil/opt.h"
#include "vif.h"

#define OFFSET(x) offsetof(VIFContext, x)
#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define OPT_RANGE_PIXEL_OFFSET (-128)

const int vif_filter1d_width1[4] = { 17, 9, 5, 3 };
const float vif_filter1d_table[4][17] = {
    { 0x1.e8a77p-8,  0x1.d373b2p-7, 0x1.9a1cf6p-6, 0x1.49fd9ep-5, 0x1.e7092ep-5, 0x1.49a044p-4, 0x1.99350ep-4, 0x1.d1e76ap-4, 0x1.e67f8p-4, 0x1.d1e76ap-4, 0x1.99350ep-4, 0x1.49a044p-4, 0x1.e7092ep-5, 0x1.49fd9ep-5, 0x1.9a1cf6p-6, 0x1.d373b2p-7, 0x1.e8a77p-8 },
    { 0x1.36efdap-6, 0x1.c9eaf8p-5, 0x1.ef4ac2p-4, 0x1.897424p-3, 0x1.cb1b88p-3, 0x1.897424p-3, 0x1.ef4ac2p-4, 0x1.c9eaf8p-5, 0x1.36efdap-6 },
    { 0x1.be5f0ep-5, 0x1.f41fd6p-3, 0x1.9c4868p-2, 0x1.f41fd6p-3, 0x1.be5f0ep-5 },
    { 0x1.54be4p-3,  0x1.55a0ep-1,  0x1.54be4p-3 }
};

static void vif_dec2(const float *src, float *dst, int src_w, int src_h,
                     int src_stride, int dst_stride)
{
    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    int i, j;

    // decimation by 2 in each direction (after gaussian blur check)
    for (i = 0; i < src_h / 2; i++) {
        for (j = 0; j < src_w / 2; j++) {
            dst[i * dst_px_stride + j] = src[(i * 2) * src_px_stride + (j * 2)];
        }
    }
}

static float vif_sum(const float *x, int w, int h, int stride)
{
    int px_stride = stride / sizeof(float);
    int i, j;

    float accum = 0;

    for (i = 0; i < h; i++) {
        float accum_inner = 0;

        for (j = 0; j < w; j++) {
            accum_inner += x[i * px_stride + j];
        }

        accum += accum_inner;
    }

    return accum;
}

static void vif_statistic(const float *mu1_sq, const float *mu2_sq,
                          const float *mu1_mu2, const float *xx_filt,
                          const float *yy_filt, const float *xy_filt,
                          float *num, float *den, int w, int h,
                          int mu1_sq_stride, int mu2_sq_stride,
                          int mu1_mu2_stride, int xx_filt_stride,
                          int yy_filt_stride, int xy_filt_stride,
                          int num_stride, int den_stride)
{
    static const float sigma_nsq = 2;
    static const float sigma_max_inv = 4.0/(255.0*255.0);

    int mu1_sq_px_stride  = mu1_sq_stride / sizeof(float);
    int mu2_sq_px_stride  = mu2_sq_stride / sizeof(float);
    int mu1_mu2_px_stride = mu1_mu2_stride / sizeof(float);
    int xx_filt_px_stride = xx_filt_stride / sizeof(float);
    int yy_filt_px_stride = yy_filt_stride / sizeof(float);
    int xy_filt_px_stride = xy_filt_stride / sizeof(float);
    int num_px_stride = num_stride / sizeof(float);
    int den_px_stride = den_stride / sizeof(float);

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
            sigma12   = xy_filt_val - mu1_mu2_val;

            if (sigma1_sq < sigma_nsq) {
                num_val = 1.0 - sigma2_sq*sigma_max_inv;
                den_val = 1.0;
            }
            else {
                sv_sq = (sigma2_sq + sigma_nsq) * sigma1_sq;
                if( sigma12 < 0 )
                {
                    num_val = 0.0;
                }
                else
                {
                    g = sv_sq - sigma12 * sigma12;
                    num_val = log2f(sv_sq / g);
                }
                den_val = log2f(1.0f + sigma1_sq / sigma_nsq);
            }

            num[i * num_px_stride + j] = num_val;
            den[i * den_px_stride + j] = den_val;
        }
    }
}

static void vif_xx_yy_xy(const float *x, const float *y, float *xx, float *yy,
                         float *xy, int w, int h, int xstride, int ystride,
                         int xxstride, int yystride, int xystride)
{
    int x_px_stride = xstride / sizeof(float);
    int y_px_stride = ystride / sizeof(float);
    int xx_px_stride = xxstride / sizeof(float);
    int yy_px_stride = yystride / sizeof(float);
    int xy_px_stride = xystride / sizeof(float);

    int i, j;

    float xval, yval, xxval, yyval, xyval;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            xval = x[i * x_px_stride + j];
            yval = y[i * y_px_stride + j];

            xxval = xval * xval;
            yyval = yval * yval;
            xyval = xval * yval;

            xx[i * xx_px_stride + j] = xxval;
            yy[i * yy_px_stride + j] = yyval;
            xy[i * xy_px_stride + j] = xyval;
        }
    }
}

static void vif_filter1d(const float *f, const float *src, float *dst,
                         float *tmpbuf, int w, int h, int src_stride,
                         int dst_stride, int fwidth, float *temp)
{

    int src_px_stride = src_stride / sizeof(float);
    int dst_px_stride = dst_stride / sizeof(float);

    float fcoeff, imgcoeff;

    int i, j, fi, fj, ii, jj;

    for (i = 0; i < h; i++) {
        /* Vertical pass. */
        for (j = 0; j < w; j++) {
            float accum = 0;

            for (fi = 0; fi < fwidth; fi++) {
                fcoeff = f[fi];

                ii = i - fwidth / 2 + fi;
                ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 1 : ii);

                imgcoeff = src[ii * src_px_stride + j];

                accum += fcoeff * imgcoeff;
            }

            temp[j] = accum;
        }

        /* Horizontal pass. */
        for (j = 0; j < w; j++) {
            float accum = 0;

            for (fj = 0; fj < fwidth; fj++) {
                fcoeff = f[fj];

                jj = j - fwidth / 2 + fj;
                jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 1 : jj);

                imgcoeff = temp[jj];

                accum += fcoeff * imgcoeff;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }
}

int compute_vif1(const float *ref, const float *dis, int w, int h,
                 int ref_stride, int dis_stride, double *score,
                 double *score_num, double *score_den, double *scores,
                 float *data_buf, float *temp)
{
    char *data_top;

    float *ref_scale;
    float *dis_scale;
    float *ref_sq;
    float *dis_sq;
    float *ref_dis;

    float *mu1;
    float *mu2;
    float *mu1_sq;
    float *mu2_sq;
    float *mu1_mu2;
    float *ref_sq_filt;
    float *dis_sq_filt;
    float *ref_dis_filt;
    float *num_array;
    float *den_array;
    float *tmpbuf;

    const float *curr_ref_scale = ref;
    const float *curr_dis_scale = dis;
    int curr_ref_stride = ref_stride;
    int curr_dis_stride = dis_stride;

    int buf_stride = ALIGN_CEIL(w * sizeof(float));
    size_t buf_sz = (size_t)buf_stride * h;

    double num = 0;
    double den = 0;

    int scale;
    int ret = 1;

    data_top = (char *) data_buf;

    ref_scale = (float *) data_top;
    data_top += buf_sz;
    dis_scale = (float *) data_top;
    data_top += buf_sz;
    ref_sq = (float *) data_top;
    data_top += buf_sz;
    dis_sq = (float *) data_top;
    data_top += buf_sz;
    ref_dis = (float *) data_top;
    data_top += buf_sz;
    mu1 = (float *) data_top;
    data_top += buf_sz;
    mu2 = (float *) data_top;
    data_top += buf_sz;
    mu1_sq = (float *) data_top;
    data_top += buf_sz;
    mu2_sq = (float *) data_top;
    data_top += buf_sz;
    mu1_mu2 = (float *) data_top;
    data_top += buf_sz;
    ref_sq_filt = (float *) data_top;
    data_top += buf_sz;
    dis_sq_filt = (float *) data_top;
    data_top += buf_sz;
    ref_dis_filt = (float *) data_top;
    data_top += buf_sz;
    num_array = (float *) data_top;
    data_top += buf_sz;
    den_array = (float *) data_top;
    data_top += buf_sz;
    tmpbuf = (float *) data_top;
    data_top += buf_sz;

    for (scale = 0; scale < 4; scale++)
    {
        const float *filter = vif_filter1d_table[scale];
        int filter_width = vif_filter1d_width1[scale];

        int buf_valid_w = w;
        int buf_valid_h = h;

        if (scale > 0)
        {
            vif_filter1d(filter, curr_ref_scale, mu1, tmpbuf, w, h,
                         curr_ref_stride, buf_stride, filter_width, temp);
            vif_filter1d(filter, curr_dis_scale, mu2, tmpbuf, w, h,
                         curr_dis_stride, buf_stride, filter_width, temp);

            vif_dec2(mu1, ref_scale, buf_valid_w, buf_valid_h, buf_stride,
                     buf_stride);
            vif_dec2(mu2, dis_scale, buf_valid_w, buf_valid_h, buf_stride,
                     buf_stride);

            w  = buf_valid_w / 2;
            h  = buf_valid_h / 2;

            buf_valid_w = w;
            buf_valid_h = h;

            curr_ref_scale = ref_scale;
            curr_dis_scale = dis_scale;

            curr_ref_stride = buf_stride;
            curr_dis_stride = buf_stride;
        }

        vif_filter1d(filter, curr_ref_scale, mu1, tmpbuf, w, h, curr_ref_stride,
                     buf_stride, filter_width, temp);
        vif_filter1d(filter, curr_dis_scale, mu2, tmpbuf, w, h, curr_dis_stride,
                     buf_stride, filter_width, temp);

        vif_xx_yy_xy(mu1, mu2, mu1_sq, mu2_sq, mu1_mu2, w, h, buf_stride,
                     buf_stride, buf_stride, buf_stride, buf_stride);

        vif_xx_yy_xy(curr_ref_scale, curr_dis_scale, ref_sq, dis_sq, ref_dis,
                     w, h, curr_ref_stride, curr_dis_stride, buf_stride,
                     buf_stride, buf_stride);

        vif_filter1d(filter, ref_sq, ref_sq_filt, tmpbuf, w, h, buf_stride,
                     buf_stride, filter_width, temp);
        vif_filter1d(filter, dis_sq, dis_sq_filt, tmpbuf, w, h, buf_stride,
                     buf_stride, filter_width, temp);
        vif_filter1d(filter, ref_dis, ref_dis_filt, tmpbuf, w, h, buf_stride,
                     buf_stride, filter_width, temp);

        vif_statistic(mu1_sq, mu2_sq, mu1_mu2, ref_sq_filt, dis_sq_filt,
                      ref_dis_filt, num_array, den_array, w, h, buf_stride,
                      buf_stride, buf_stride, buf_stride, buf_stride,
                      buf_stride, buf_stride, buf_stride);

        num = vif_sum(num_array, buf_valid_w, buf_valid_h, buf_stride);
        den = vif_sum(den_array, buf_valid_w, buf_valid_h, buf_stride);

        scores[2*scale] = num;
        scores[2*scale+1] = den;
    }

    *score_num = 0.0;
    *score_den = 0.0;
    for (scale = 0; scale < 4; ++scale)
    {
        *score_num += scores[2*scale];
        *score_den += scores[2*scale+1];
    }
    if (*score_den == 0.0)
    {
        *score = 1.0f;
    }
    else
    {
        *score = (*score_num) / (*score_den);
    }

    ret = 0;

    return ret;
}
