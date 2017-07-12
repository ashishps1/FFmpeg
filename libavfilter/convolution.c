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

#include <stdbool.h>
#include "convolution.h"

#define FORCE_INLINE __attribute__((always_inline))
#define RESTRICT __restrict

static inline int floorn(int n, int m)
{
    return n - n % m;
}

static inline int ceiln(int n, int m)
{
    return n % m ? n + (m - n % m) : n;
}

FORCE_INLINE inline float convolution_edge(bool horizontal, const float *filter,
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
            if (j_tap < 0)
                j_tap = -j_tap;
            else if (j_tap >= w)
                j_tap = w - (j_tap - w + 1);
        } else {
            if (i_tap < 0)
                i_tap = -i_tap;
            else if (i_tap >= h)
                i_tap = h - (i_tap - h + 1);
        }

        accum += filter[k] * src[i_tap * stride + j_tap];
    }
    return accum;
}

static void convolution_x_c(const float *filter, int filt_width,
                            const float *src, float *dst, int w, int h,
                            int src_stride, int dst_stride, int step)
{
    int radius = filt_width / 2;
    int borders_left = ceiln(radius, step);
    int borders_right = floorn(w - (filt_width - radius), step);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < borders_left; j += step) {
            dst[i * dst_stride + j / step] = convolution_edge(true, filter,
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
            dst[i * dst_stride + j / step] = convolution_edge(true, filter,
                                                              filt_width, src,
                                                              w, h, src_stride,
                                                              i, j);
        }
    }
}

static void convolution_y_c(const float *filter, int filt_width,
                            const float *src, float *dst, int w, int h,
                            int src_stride, int dst_stride, int step)
{
    int radius = filt_width / 2;
    int borders_top = ceiln(radius, step);
    int borders_bottom = floorn(h - (filt_width - radius), step);

    for (int i = 0; i < borders_top; i += step) {
        for (int j = 0; j < w; j++) {
            dst[(i / step) * dst_stride + j] = convolution_edge(false, filter,
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
            dst[(i / step) * dst_stride + j] = convolution_edge(false, filter,
                                                                filt_width, src,
                                                                w, h, src_stride,
                                                                i, j);
        }
    }
}

void convolution_f32_c(const float *filter, int filt_width, const float *src,
                       float *dst, float *tmp, int w, int h, int src_stride,
                       int dst_stride)
{
    convolution_y_c(filter, filt_width, src, tmp, w, h, src_stride,
                    dst_stride, 1);
    convolution_x_c(filter, filt_width, tmp, dst, w, h, src_stride,
                    dst_stride, 1);
}

