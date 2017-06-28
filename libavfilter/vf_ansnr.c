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
 * Calculate the ANSNR between two input videos.
 */

#include <inttypes.h>
#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "dualinput.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

typedef struct ANSNRContext {
    const AVClass *class;
    FFDualInputContext dinput;
    int width;
    int height;
    char *format;
    double ansnr_sum;
    uint64_t nb_frames;
    int nb_components;
} ANSNRContext;

#define OFFSET(x) offsetof(ANSNRContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM
#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define OPT_RANGE_PIXEL_OFFSET (-128)

static const AVOption ansnr_options[] = {
    { NULL }
};

AVFILTER_DEFINE_CLASS(ansnr);


const float ansnr_filter1d_ref_s[3] = { 0x1.00243ap-2, 0x1.ffb78cp-2, 0x1.00243ap-2 };

const float ansnr_filter1d_dis_s[5] = { 0x1.be5f0ep-5, 0x1.f41fd6p-3, 0x1.9c4868p-2, 0x1.f41fd6p-3, 0x1.be5f0ep-5 };

const int ansnr_filter1d_ref_width = 3;
const int ansnr_filter1d_dis_width = 5;

const float ansnr_filter2d_ref_s[3*3] = {
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0,
    2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0,
    1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0
};

const float ansnr_filter2d_dis_s[5*5] = {
    2.0 / 571.0,  7.0 / 571.0,  12.0 / 571.0,  7.0 / 571.0,  2.0 / 571.0,
    7.0 / 571.0,  31.0 / 571.0, 52.0 / 571.0,  31.0 / 571.0, 7.0 / 571.0,
    12.0 / 571.0, 52.0 / 571.0, 127.0 / 571.0, 52.0 / 571.0, 12.0 / 571.0,
    7.0 / 571.0,  31.0 / 571.0, 52.0 / 571.0,  31.0 / 571.0, 7.0 / 571.0,
    2.0 / 571.0,  7.0 / 571.0,  12.0 / 571.0,  7.0 / 571.0,  2.0 / 571.0
};

const int ansnr_filter2d_ref_width = 3;
const int ansnr_filter2d_dis_width = 5;

static int offset_image(int16_t *buf, int off, int width, int height, int stride)
{
	char *byte_ptr = (char *)buf;
	int ret = 1;
	int i, j;

	for (i = 0; i < height; ++i)
	{
		int16_t *row_ptr = (int16_t *)byte_ptr;

		for (j = 0; j < width; ++j)
		{
			row_ptr[j] += off;
			printf("%d ",(int)row_ptr[j]);
		}
        printf("\n");
		byte_ptr += stride;
	}

	ret = 0;

	return ret;
}

static void ansnr_mse(const int16_t *ref, const int16_t *dis, float *signal,
                      float *noise, int w, int h, int ref_stride,
                      int dis_stride)
{
    int ref_px_stride = ref_stride / sizeof(int16_t);
    int dis_px_stride = dis_stride / sizeof(int16_t);
    int i, j;

    float ref_val, dis_val;

    float signal_accum = 0;
    float noise_accum = 0;

    for (i = 0; i < h; ++i) {
        float signal_accum_inner = 0;
        float noise_accum_inner = 0;

        for (j = 0; j < w; ++j) {
            ref_val = ref[i * ref_px_stride + j];
            dis_val = dis[i * dis_px_stride + j];

            signal_accum_inner   += ref_val * ref_val;
            noise_accum_inner += (ref_val - dis_val) * (ref_val - dis_val);
        }

        signal_accum   += signal_accum_inner;
        noise_accum += noise_accum_inner;
    }

    if (signal)
        *signal = signal_accum;
    if (noise)
        *noise = noise_accum;
}

static void ansnr_filter2d(const int16_t *f, const int16_t *src, int16_t *dst,
                           int w, int h, int src_stride, int dst_stride,
                           int fwidth)
{
    int src_px_stride = src_stride / sizeof(int16_t);
    int dst_px_stride = dst_stride / sizeof(int16_t);

    float fcoeff, imgcoeff;
    int i, j, fi, fj, ii, jj;

    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            float accum = 0;

            for (fi = 0; fi < fwidth; ++fi) {
                float accum_inner = 0;

                for (fj = 0; fj < fwidth; ++fj) {
                    fcoeff = f[fi * fwidth + fj];

                    ii = i - fwidth / 2 + fi;
                    jj = j - fwidth / 2 + fj;

                    if (ii < 0) ii = -ii;
                    else if (ii >= h) ii = 2 * h - ii - 1;
                    if (jj < 0) jj = -jj;
                    else if (jj >= w) jj = 2 * w - jj - 1;
                    imgcoeff = src[ii * src_px_stride + jj];
                    accum_inner += fcoeff * imgcoeff;
                }

                accum += accum_inner;
            }

            dst[i * dst_px_stride + j] = accum;
        }
    }
}

static int compute_ansnr(const int16_t *ref, const int16_t *dis, int w,
                         int h, int ref_stride, int dis_stride, double *score,
                         double *score_psnr, double peak, double psnr_max)
{
    int16_t *ptr = ref;
    static int pt = 0;
    
    printf("\n------------------------\n");
    if(pt<10){
        int i,j;
        for(i=0;i<h;i++){
            for(j=0;j<w;j++){
              //  printf("%d ",(int)ptr[j]);
            }
            ptr += ref_stride;
            //printf("\n");
        }
        pt++;    
    }
    
    int16_t *data_buf = 0;
    char *data_top;

    int16_t *ref_filtr;
    int16_t *ref_filtd;
    int16_t *dis_filtd;

    float signal, noise;

    int buf_stride = ALIGN_CEIL(w * sizeof(int16_t));
    size_t buf_sz_one = (size_t)buf_stride * h;

    int ret = 1;

    if (SIZE_MAX / buf_sz_one < 3)
    {
        return AVERROR(EINVAL);
    }

    if (!(data_buf = av_malloc(buf_sz_one * 3)))
    {
        return AVERROR(EINVAL);
    }

    data_top = (char *)data_buf;

    ref_filtr = (int16_t *)data_top; data_top += buf_sz_one;
    ref_filtd = (int16_t *)data_top; data_top += buf_sz_one;
    dis_filtd = (int16_t *)data_top; data_top += buf_sz_one;

    ansnr_filter2d(ansnr_filter2d_ref_s, ref, ref_filtr, w, h, ref_stride, buf_stride, ansnr_filter2d_ref_width);
    ansnr_filter2d(ansnr_filter2d_dis_s, ref, ref_filtd, w, h, ref_stride, buf_stride, ansnr_filter2d_dis_width);
    ansnr_filter2d(ansnr_filter2d_dis_s, dis, dis_filtd, w, h, dis_stride, buf_stride, ansnr_filter2d_dis_width);

    ansnr_mse(ref_filtr, dis_filtd, &signal, &noise, w, h, buf_stride, buf_stride);

    *score = noise==0 ? psnr_max : 10.0 * log10(signal / noise);
    
    double eps = 1e-10;
    *score_psnr = FFMIN(10 * log10(peak * peak * w * h / FFMAX(noise, eps)), psnr_max);

    ret = 0;

    return ret;
}

static AVFrame *do_ansnr(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    ANSNRContext *s = ctx->priv;

    char *format = s->format;

    double max_psnr;

    if (!strcmp(format, "yuv420p") || !strcmp(format, "yuv422p") || !strcmp(format, "yuv444p")) {
        max_psnr = 60;
    }
    else if (!strcmp(format, "yuv420p10le") || !strcmp(format, "yuv422p10le") || !strcmp(format, "yuv444p10le")) {
        max_psnr = 72;
    }

    double score = 0.0;

    int w = s->width;
    int h = s->height;

    double stride;

    stride = ALIGN_CEIL(w * sizeof(int16_t));

    double score_psnr = 0;
    
    int16_t *ref_data = ref->data[0];
    int16_t *main_data = main->data[0]; 
    
    offset_image(ref_data, OPT_RANGE_PIXEL_OFFSET, w, h, stride);
    offset_image(main_data, OPT_RANGE_PIXEL_OFFSET, w, h, stride);    

    compute_ansnr(ref_data, main_data, w, h, stride, stride, &score, &score_psnr, 255.75, max_psnr);

    s->nb_frames++;
    
    printf("ansnr: %.3f\n", score);
    
    s->ansnr_sum += score;

    return main;
}

static av_cold int init(AVFilterContext *ctx)
{
    ANSNRContext *s = ctx->priv;

    s->dinput.process = do_ansnr;

    return 0;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV420P10LE,
        AV_PIX_FMT_NONE
    };

    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}


static int config_input_ref(AVFilterLink *inlink)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    AVFilterContext *ctx  = inlink->dst;
    ANSNRContext *s = ctx->priv;
    s->nb_components = desc->nb_components;
    if (ctx->inputs[0]->w != ctx->inputs[1]->w ||
        ctx->inputs[0]->h != ctx->inputs[1]->h) {
        av_log(ctx, AV_LOG_ERROR, "Width and height of input videos must be same.\n");
        return AVERROR(EINVAL);
    }
    if (ctx->inputs[0]->format != ctx->inputs[1]->format) {
        av_log(ctx, AV_LOG_ERROR, "Inputs must be of same pixel format.\n");
        return AVERROR(EINVAL);
    }
    
    s->width = ctx->inputs[0]->w;
    s->height = ctx->inputs[0]->h;
    s->format = av_get_pix_fmt_name(ctx->inputs[0]->format);

    return 0;
}


static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    ANSNRContext *s = ctx->priv;
    AVFilterLink *mainlink = ctx->inputs[0];
    int ret;

    outlink->w = mainlink->w;
    outlink->h = mainlink->h;
    outlink->time_base = mainlink->time_base;
    outlink->sample_aspect_ratio = mainlink->sample_aspect_ratio;
    outlink->frame_rate = mainlink->frame_rate;
    if ((ret = ff_dualinput_init(ctx, &s->dinput)) < 0)
        return ret;

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *inpicref)
{
    ANSNRContext *s = inlink->dst->priv;
    return ff_dualinput_filter_frame(&s->dinput, inlink, inpicref);
}

static int request_frame(AVFilterLink *outlink)
{
    ANSNRContext *s = outlink->src->priv;
    return ff_dualinput_request_frame(&s->dinput, outlink);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    ANSNRContext *s = ctx->priv;

    ff_dualinput_uninit(&s->dinput);
}

static const AVFilterPad ansnr_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },{
        .name         = "reference",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_input_ref,
    },
    { NULL }
};

static const AVFilterPad ansnr_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
        .request_frame = request_frame,
    },
    { NULL }
};

AVFilter ff_vf_ansnr = {
    .name          = "ansnr",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the PSNR between two video streams."),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(ANSNRContext),
    .priv_class    = &ansnr_class,
    .inputs        = ansnr_inputs,
    .outputs       = ansnr_outputs,
};
