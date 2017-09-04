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

#ifndef MOTION_TOOLS_H_
#define MOTION_TOOLS_H_

#define N 15

#include <stddef.h>
#include <stdint.h>

typedef struct VMAFMotionDSPContext {
    uint64_t (*image_sad)(const uint16_t *img1, const uint16_t *img2, int w, int h,
                            ptrdiff_t img1_stride, ptrdiff_t img2_stride);
} VMAFMotionDSPContext;

void ff_vmafmotion_init_x86(VMAFMotionDSPContext *dsp);

static const float FILTER_5[5] = {
    0.054488685,
    0.244201342,
    0.402619947,
    0.244201342,
    0.054488685
};

void convolution_f32(const int *filter, int filt_width, const void *src,
                     uint16_t *dst, uint16_t *tmp, int w, int h,
                     ptrdiff_t src_stride, ptrdiff_t dst_stride, uint8_t type);

int compute_vmafmotion(const uint16_t *ref, const uint16_t *main, int w, int h,
                       ptrdiff_t ref_stride, ptrdiff_t main_stride, double *score);

#endif /* MOTION_TOOLS_H_ */
