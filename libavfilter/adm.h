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

#ifndef AVFILTER_ADM_H
#define AVFILTER_ADM_H

#define VIEW_DIST 3.0f
#define REF_DISPLAY_HEIGHT 1080
/** Percentage of frame to discard on all 4 sides */
#define ADM_BORDER_FACTOR (0.1)

typedef struct adm_dwt_band_t {
    float *band_a; /** Low-pass V + low-pass H. */
    float *band_v; /** Low-pass V + high-pass H. */
    float *band_h; /** High-pass V + low-pass H. */
    float *band_d; /** High-pass V + high-pass H. */
} adm_dwt_band_t;

/**
 * The following dwt visibility threshold parameters are taken from
 * "Visibility of Wavelet Quantization Noise"
 * by A. B. Watson, G. Y. Yang, J. A. Solomon and J. Villasenor
 * IEEE Trans. on Image Processing, Vol. 6, No 8, Aug. 1997
 * Page 1170, formula (7) and corresponding Table IV
 * Table IV has 2 entries for Cb and Cr thresholds
 * Chose those corresponding to subject "sfl" since they are lower
 * These thresholds were obtained and modeled for the 7-9 biorthogonal wavelet basis
 */
struct dwt_model_params {
    float a;
    float k;
    float f0;
    float g[4];
};

static const float dwt2_db2_coeffs_lo[4] = { 0.482962913144690, 0.836516303737469, 0.224143868041857, -0.129409522550921 };
static const float dwt2_db2_coeffs_hi[4] = { -0.129409522550921, -0.224143868041857, 0.836516303737469, -0.482962913144690 };

/** 0 -> Y, 1 -> Cb, 2 -> Cr */
static const struct dwt_model_params dwt_7_9_YCbCr_threshold[3] = {
    { .a = 0.495, .k = 0.466, .f0 = 0.401, .g = { 1.501, 1.0, 0.534, 1.0} },
    { .a = 1.633, .k = 0.353, .f0 = 0.209, .g = { 1.520, 1.0, 0.502, 1.0} },
    { .a = 0.944, .k = 0.521, .f0 = 0.404, .g = { 1.868, 1.0, 0.516, 1.0} }
};

/**
 * The following dwt basis function amplitudes, A(lambda,theta), are taken from
 * "Visibility of Wavelet Quantization Noise"
 * by A. B. Watson, G. Y. Yang, J. A. Solomon and J. Villasenor
 * IEEE Trans. on Image Processing, Vol. 6, No 8, Aug. 1997
 * Page 1172, Table V
 * The table has been transposed, i.e. it can be used directly to obtain A[lambda][theta]
 * These amplitudes were calculated for the 7-9 biorthogonal wavelet basis
 */
static const float dwt_7_9_basis_function_amplitudes[6][4] = {
    { 0.62171,  0.67234,  0.72709,  0.67234  },
    { 0.34537,  0.41317,  0.49428,  0.41317  },
    { 0.18004,  0.22727,  0.28688,  0.22727  },
    { 0.091401, 0.11792,  0.15214,  0.11792  },
    { 0.045943, 0.059758, 0.077727, 0.059758 },
    { 0.023013, 0.030018, 0.039156, 0.030018 }
};

int compute_adm1(const float *ref, const float *dis, int w, int h,
                 int ref_stride, int dis_stride, double *score,
                 double *score_num, double *score_den, double *scores,
                 float *data_buf, float *temp_lo, float* temp_hi);

#endif /* AVFILTER_ADM_H */
