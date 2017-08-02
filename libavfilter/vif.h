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

#ifndef AVFILTER_VIF_H
#define AVFILTER_VIF_H

#define N 9

static const int vif_filter_width[4] = { 17, 9, 5, 3 };

static const float vif_filter_table[4][17] = {
    { 0.007456, 0.014266, 0.025031, 0.040282, 0.059453, 0.080475, 0.099904,
      0.113746, 0.118774, 0.113746, 0.099904, 0.080475, 0.059453, 0.040282,
      0.025031, 0.014266, 0.007456 },
    { 0.018978, 0.055898, 0.120921, 0.192116, 0.224174, 0.192116, 0.120921,
      0.055898, 0.018978 },
    { 0.054489, 0.244201, 0.402620, 0.244201, 0.054489 },
    { 0.166378, 0.667243, 0.166378 }
};

int compute_vif2(const float *ref, const float *main, int w, int h,
                 int ref_stride, int main_stride, double *score,
                 double *score_num, double *score_den, double *scores,
                 float *data_buf, float *temp);

#endif /* AVFILTER_VIF_H */
