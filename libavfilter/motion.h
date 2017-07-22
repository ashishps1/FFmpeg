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

static const float FILTER_5[5] = {
    0.054488685,
    0.244201342,
    0.402619947,
    0.244201342,
    0.054488685
};

void convolution_f32(const float *filter, int filt_width, const float *src,
                     float *dst, float *tmp, int w, int h, int src_stride,
                     int dst_stride);

int compute_motion1(const float *ref, const float *dis, int w, int h,
                    int ref_stride, int dis_stride, double *score);

#endif /* MOTION_TOOLS_H_ */
