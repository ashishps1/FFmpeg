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
 * Caculate the VMAF between two input videos.
 */

#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <inttypes.h>
#include <pthread.h>
#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "dualinput.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "video.h"
#include "vmaf.h"
#include "libvmaf.h"

typedef struct VMAFContext {
    const AVClass *class;
    FFDualInputContext dinput;
    uint64_t nb_frames;
    FILE *stats_file;
    char *stats_file_str;
    int stats_version;
    int stats_header_written;
    int stats_add_max;
    int is_rgb;
	pthread_t thread;
	pthread_t main_thread_id;
	pthread_t vmaf_thread_id;
	pthread_attr_t attr;
    uint8_t rgba_map[4];
    char comps[4];
    int nb_components;
    int planewidth[4];
    int planeheight[4];
    double vmaf_sum;
    double planeweight[4];
    VMAFDSPContext dsp;
} VMAFContext;


static char *format;
static int width, height;
pthread_mutex_t lock;
pthread_cond_t cond;
int get_next = 0;
int data_yes = 0;
AVFrame *gmain;
AVFrame *gref;

#define OFFSET(x) offsetof(VMAFContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption vmaf_options[] = {
    {"stats_file", "Set file where to store per-frame difference information", OFFSET(stats_file_str), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS },
    {"f",          "Set file where to store per-frame difference information", OFFSET(stats_file_str), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS },
    {"stats_version", "Set the format version for the stats file.",               OFFSET(stats_version),  AV_OPT_TYPE_INT,    {.i64=1},    1, 2, FLAGS },
    {"output_max",  "Add raw stats (max values) to the output log.",            OFFSET(stats_add_max), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    { NULL }
};

AVFILTER_DEFINE_CLASS(vmaf);

static inline double get_vmaf(double vmaf_sum, uint64_t nb_frames)
{
    return vmaf_sum/nb_frames;
}

static void set_meta(AVDictionary **metadata, const char *key, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%0.7f", d);
    av_dict_set(metadata,key,value,0);
}

static void read_frame(uint8_t *ref_buf, int *ref_stride, uint8_t *main_buf, int *main_stride){
	pthread_mutex_lock(&lock);
	while(gref == NULL){
		pthread_cond_wait(&cond, &lock);
	}
	
    *ref_stride = gref->linesize[0];
    *main_stride = gmain->linesize[0];

	ref_buf = malloc(sizeof(uint8_t));
	main_buf = malloc(sizeof(uint8_t));

	memcpy(ref_buf, gref->data[0], sizeof(uint8_t));
	memcpy(main_buf, gmain->data[0], sizeof(uint8_t));	
	
	gref = NULL;
	gmain = NULL;

	pthread_cond_signal(&cond);    
	pthread_mutex_unlock(&lock);
}

static AVFrame *do_vmaf(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    VMAFContext *s = ctx->priv;

    AVDictionary **metadata = avpriv_frame_get_metadatap(main);

	pthread_mutex_lock(&lock);
	while(gref != NULL){
		pthread_cond_wait(&cond, &lock);
	}

	gref = malloc(sizeof(AVFrame));
	gmain = malloc(sizeof(AVFrame));

	memcpy(gref, ref, sizeof(AVFrame));
	memcpy(gmain, main, sizeof(AVFrame));	

	pthread_cond_signal(&cond);    
	pthread_mutex_unlock(&lock);

/*
	double score = 0;
    //set_meta(metadata, "lavfi.vmaf.score.",score);
    //av_log(ctx, AV_LOG_INFO, "vmaf score for frame %lu is %lf.\n",s->nb_frames,score);
	//printf("v");
	static int p = 0;
	p = 1;
    s->vmaf_sum += score;

    s->nb_frames++;
*/
    return main;
}

static void compute_vmaf_score()
{
	printf("inside compute vmaf6\n");
    
	printf("width=%d,height=%d,format=%s\n",width,height,format);

    char *model_path = "/usr/local/share/model/vmaf_v0.6.1.pkl";

    double vmaf_score;

	vmaf_score = compute_vmaf(format, width, height, read_frame, model_path);
	//printf("compute vmaf completed\n");
    return vmaf_score;
}


static void *BusyWork(void *t)
 {
    int i;
    long tid;
    double result=0.0;
    tid = (long)t;
	printf("busy\n");
	compute_vmaf_score();
    pthread_exit((void*) t);
 }


static av_cold int init(AVFilterContext *ctx)
{
    VMAFContext *s = ctx->priv;
	
    if (s->stats_file_str) {
        if (s->stats_version < 2 && s->stats_add_max) {
            av_log(ctx, AV_LOG_ERROR,
                   "stats_add_max was specified but stats_version < 2.\n" );
            return AVERROR(EINVAL);
        }
        if (!strcmp(s->stats_file_str, "-")) {
            s->stats_file = stdout;
        } else {
            s->stats_file = fopen(s->stats_file_str, "w");
            if (!s->stats_file) {
                int err = AVERROR(errno);
                char buf[128];
                av_strerror(err, buf, sizeof(buf));
                av_log(ctx, AV_LOG_ERROR, "Could not open stats file %s: %s\n",
                       s->stats_file_str, buf);
                return err;
            }
        }
    }

    s->dinput.process = do_vmaf;
    return 0;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV420P,
        AV_PIX_FMT_YUV444P10LE, AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_YUV420P10LE,
        AV_PIX_FMT_NONE
    };
	printf("query formats\n");
    VMAFContext *s = ctx->priv;

    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}


static int config_input_ref(AVFilterLink *inlink)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    AVFilterContext *ctx  = inlink->dst;
    VMAFContext *s = ctx->priv;
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

	printf("inside config input ref\n");

	format = av_get_pix_fmt_name(ctx->inputs[0]->format);
	width = ctx->inputs[0]->w;
	height = ctx->inputs[0]->h;

	/* Initialize mutex and condition variable objects */
    pthread_mutex_init(&lock, NULL);
    pthread_cond_init (&cond, NULL);

	pthread_attr_init(&s->attr);
	int d = 5;
		
	s->main_thread_id=pthread_self();
	
	int rc = pthread_create(&s->thread, &s->attr, BusyWork, (void *)d);
	s->vmaf_thread_id = s->thread;	


    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    VMAFContext *s = ctx->priv;
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
    VMAFContext *s = inlink->dst->priv;
    return ff_dualinput_filter_frame(&s->dinput, inlink, inpicref);
}

static int request_frame(AVFilterLink *outlink)
{
    VMAFContext *s = outlink->src->priv;
    return ff_dualinput_request_frame(&s->dinput, outlink);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    VMAFContext *s = ctx->priv;

    if (s->nb_frames > 0) {
        av_log(ctx, AV_LOG_INFO, "VMAF average:%f\n",
               get_vmaf(s->vmaf_sum, s->nb_frames));
    }
    ff_dualinput_uninit(&s->dinput);

    if (s->stats_file && s->stats_file != stdout)
        fclose(s->stats_file);

	return 0;
}

static const AVFilterPad vmaf_inputs[] = {
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

static const AVFilterPad vmaf_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
        .request_frame = request_frame,
    },
    { NULL }
};

AVFilter ff_vf_vmaf = {
    .name          = "vmaf",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the VMAF between two video streams."),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .priv_size     = sizeof(VMAFContext),
    .priv_class    = &vmaf_class,
    .inputs        = vmaf_inputs,
    .outputs       = vmaf_outputs,
};
