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

#define N 15

static const int vif_filter_width[4] = { 17, 9, 5, 3 };

static const float vif_filter_table[4][17] = {
    { 0.00745626912, 0.0142655009, 0.0250313189, 0.0402820669, 0.0594526194,
      0.0804751068,  0.0999041125, 0.113746084,  0.118773937,  0.113746084,
      0.0999041125,  0.0804751068, 0.0594526194, 0.0402820669, 0.0250313189,
      0.0142655009,  0.00745626912 },
    { 0.0189780835,  0.0558981746, 0.120920904,  0.192116052, 0.224173605,
      0.192116052,   0.120920904,  0.0558981746, 0.0189780835 },
    { 0.054488685,   0.244201347,  0.402619958,  0.244201347, 0.054488685 },
    { 0.166378498,   0.667243004,  0.166378498 }
};

int compute_vif2(const int filter[4][17], const void *ref, const void *main, int w, int h,
                 ptrdiff_t ref_stride, ptrdiff_t main_stride, double *score,
                 double *score_num, double *score_den, double *scores,
                 uint64_t *data_buf, uint64_t *temp, int type);

#endif /* AVFILTER_VIF_H */
