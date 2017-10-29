/*
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

typedef struct ADMData {
    int width;
    int height;
    const AVPixFmtDescriptor *desc;
    int16_t *data_buf;
    int16_t *temp_lo;
    int16_t *temp_hi;
    double adm_sum;
    uint64_t nb_frames;
} ADMData;

int ff_adm_init(ADMData *data, int w, int h, enum AVPixelFormat fmt);
double ff_adm_process(ADMData *data, AVFrame *ref, AVFrame *main, double *score,
                      double *score_num, double *score_den, double *scores);
double ff_adm_uninit(ADMData *data);

#endif /* AVFILTER_ADM_H */
