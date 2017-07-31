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
    { 0x1.e8a77p-8,  0x1.d373b2p-7, 0x1.9a1cf6p-6, 0x1.49fd9ep-5, 0x1.e7092ep-5,
      0x1.49a044p-4, 0x1.99350ep-4, 0x1.d1e76ap-4, 0x1.e67f8p-4,  0x1.d1e76ap-4,
      0x1.99350ep-4, 0x1.49a044p-4, 0x1.e7092ep-5, 0x1.49fd9ep-5, 0x1.9a1cf6p-6,
      0x1.d373b2p-7, 0x1.e8a77p-8 },
    { 0x1.36efdap-6, 0x1.c9eaf8p-5, 0x1.ef4ac2p-4, 0x1.897424p-3, 0x1.cb1b88p-3,
      0x1.897424p-3, 0x1.ef4ac2p-4, 0x1.c9eaf8p-5, 0x1.36efdap-6 },
    { 0x1.be5f0ep-5, 0x1.f41fd6p-3, 0x1.9c4868p-2, 0x1.f41fd6p-3, 0x1.be5f0ep-5 },
    { 0x1.54be4p-3,  0x1.55a0ep-1,  0x1.54be4p-3 }
};

int compute_vif2(const int vif_filter[4][17], const uint64_t *ref, const uint64_t *main, int w, int h,
                 int ref_stride, int main_stride, double *score,
                 double *score_num, double *score_den, double *scores,
                 uint64_t *data_buf, uint64_t *temp);

#endif /* AVFILTER_VIF_H */
