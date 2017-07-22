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

#include "libavutil/opt.h"
#include "motion.h"

#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define OPT_RANGE_PIXEL_OFFSET (-128)

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

void convolution_f32(const float *filter, int filt_width, const float *src,
                     float *dst, float *tmp, int w, int h, int src_stride,
                     int dst_stride)
{
    convolution_y(filter, filt_width, src, tmp, w, h, src_stride,
                  dst_stride, 1);
    convolution_x(filter, filt_width, tmp, dst, w, h, src_stride,
                  dst_stride, 1);
}

int compute_motion1(const float *ref, const float *dis, int w, int h,
                    int ref_stride, int dis_stride, double *score)
{
    *score = image_sad(ref, dis, w, h, ref_stride / sizeof(float),
                       dis_stride / sizeof(float));

    return 0;
}

