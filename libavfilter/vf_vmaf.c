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
 * Calculate the VMAF between two input videos.
 */

#include <locale.h>
#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "dualinput.h"
#include "drawutils.h"
#include "formats.h"
#include "internal.h"
#include "video.h"
#include "adm.h"
#include "vmaf_motion.h"
#include "vif.h"
#include "vmaf.h"

typedef struct VMAFContext {
    const AVClass *class;
    FFDualInputContext dinput;
    const AVPixFmtDescriptor *desc;
    int width;
    int height;
    uint8_t called;
    double score;
    double scores[8];
    double score_num;
    double score_den;
    int conv_filter[5];
    float *ref_data;
    float *main_data;
    float *adm_data_buf;
    float *adm_temp_lo;
    float *adm_temp_hi;
    uint16_t *prev_blur_data;
    uint16_t *blur_data;
    uint16_t *temp_data;
    float *vif_data_buf;
    float *vif_temp;
    double prev_motion_score;
    double vmaf_score;
    uint64_t nb_frames;
    char *model_path;
    char *log_path;
    char *log_fmt;
    int enable_transform;
    int phone_model;
    char *pool;
    DArray adm_array,
           adm_scale_array[4],
           motion_array,
           motion2_array,
           vif_scale_array[4],
           vif_array;
} VMAFContext;

#define OFFSET(x) offsetof(VMAFContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption vmaf_options[] = {
    {"model_path",  "Set the model to be used for computing vmaf.",                     OFFSET(model_path), AV_OPT_TYPE_STRING, {.str="libavfilter/data/vmaf_v0.6.1.pkl.model"}, 0, 1, FLAGS},
    {"log_path",  "Set the file path to be used to store logs.",                        OFFSET(log_path), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 1, FLAGS},
    {"log_fmt",  "Set the format of the log (xml or json).",                            OFFSET(log_fmt), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 1, FLAGS},
    {"enable_transform",  "Enables transform for computing vmaf.",                      OFFSET(enable_transform), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"phone_model",  "Invokes the phone model that will generate higher VMAF scores.",  OFFSET(phone_model), AV_OPT_TYPE_BOOL, {.i64=0}, 0, 1, FLAGS},
    {"pool",  "Set the pool method to be used for computing vmaf.",                     OFFSET(pool), AV_OPT_TYPE_STRING, {.str="mean"}, 0, 1, FLAGS},
    { NULL }
};

AVFILTER_DEFINE_CLASS(vmaf);

#define MAX_ALIGN 32
#define ALIGN_CEIL(x) ((x) + ((x) % MAX_ALIGN ? MAX_ALIGN - (x) % MAX_ALIGN : 0))
#define INIT_FRAMES 1000

const char *norm_type = "linear_rescale";

const double score_clip[2] = {
    0.0,
    100.0
};

const char *feature_names[6] = {
    "VMAF_feature_adm2_score",
    "VMAF_feature_motion2_score",
    "VMAF_feature_vif_scale0_score",
    "VMAF_feature_vif_scale1_score",
    "VMAF_feature_vif_scale2_score",
    "VMAF_feature_vif_scale3_score"
};

const double intercepts[7] = {
    -0.3092981927591963,
    -1.7993968597186747,
    -0.003017198086831897,
    -0.1728125095425364,
    -0.5294309090081222,
    -0.7577185792093722,
    -1.083428597549764
};

const char *model_type = "LIBSVMNUSVR";

const double slopes[7] = {
    0.012020766332648465,
    2.8098077502505414,
    0.06264407466686016,
    1.222763456258933,
    1.5360318811084146,
    1.7620864995501058,
    2.08656468286432
};

const double score_transform[3] = {
    1.70674692,
    1.72643844,
    -0.00705305
};

void init_arr(DArray *a, size_t init_size)
{
    a->array = (double *) av_malloc(init_size * sizeof(double));
    a->used = 0;
    a->size = init_size;
}

void append_arr(DArray *a, double e)
{
    if (a->used == a->size) {
        a->size *= 2;
        a->array = (double *) av_realloc(a->array, a->size * sizeof(double));
    }
    a->array[a->used++] = e;
}

double get_at_pos(DArray *a, int pos)
{
    return a->array[pos];
}

void free_arr(DArray *a)
{
    av_free(a->array);
    a->array = NULL;
    a->used = a->size = 0;
}

extern int libsvm_version;

typedef struct {
    int index;
    double value;
} svm_node;

typedef struct {
    int l;
    double *y;
    svm_node **x;
} svm_problem;

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };    /** svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /** kernel_type */

typedef struct {
    int svm_type;
    int kernel_type;
    int degree;    /** for poly */
    double gamma;    /** for poly/rbf/sigmoid */
    double coef0;    /** for poly/sigmoid */

    /** these are for training only */
    double cache_size; /** in MB */
    double eps;    /** stopping criteria */
    double C;    /** for C_SVC, EPSILON_SVR and NU_SVR */
    int nr_weight;        /** for C_SVC */
    int *weight_label;    /** for C_SVC */
    double* weight;        /** for C_SVC */
    double nu;    /** for NU_SVC, ONE_CLASS, and NU_SVR */
    double p;    /** for EPSILON_SVR */
    int shrinking;    /** use the shrinking heuristics */
    int probability; /** do probability estimates */
} svm_parameter;

/**
 * svm_model
 */
typedef struct {
    svm_parameter param;    /** parameter */
    int nr_class;        /** number of classes, = 2 in regression/one class svm */
    int l;            /** total #SV */
    svm_node **SV;        /** SVs (SV[l]) */
    double **sv_coef;    /** coefficients for SVs in decision functions (sv_coef[k-1][l]) */
    double *rho;        /** constants in decision functions (rho[k*(k-1)/2]) */
    double *probA;        /** pariwise probability information */
    double *probB;
    int *sv_indices;        /** sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

    /** for classification only */

    int *label;        /** label of each class (label[k]) */
    int *nSV;        /** number of SVs for each class (nSV[k]) */
    /** nSV[0] + nSV[1] + ... + nSV[k-1] = l */
    int free_sv;        /** 1 if svm_model is created by svm_load_model*/
    /** 0 if svm_model is created by svm_train */
} svm_model;

#define swap(type, x, y) { type t=x; x=y; y=t; }

static inline double power(double base, int times)
{
    double tmp = base, ret = 1.0;

    for(int t = times; t > 0; t /= 2) {
        if(t % 2 == 1) {
            ret *= tmp;
        }
        tmp = tmp * tmp;
    }
    return ret;
}

typedef struct {
    const svm_node **x;
    double *x_square;

    // svm_parameter
    const int kernel_type;
    const int degree;
    const double gamma;
    const double coef0;
} Kernel;

static double dot(const svm_node *px, const svm_node *py)
{
    double sum = 0;
    while(px->index != -1 && py->index != -1) {
        if(px->index == py->index) {
            sum += px->value * py->value;
            px++;
            py++;
        } else {
            if(px->index > py->index) {
                py++;
            } else {
                px++;
            }
        }
    }
    return sum;
}

static double k_function(const svm_node *x, const svm_node *y,
                         const svm_parameter *param)
{
    switch(param->kernel_type)
    {
        case LINEAR:
            return dot(x, y);
        case POLY:
            return power(param->gamma * dot(x, y) + param->coef0, param->degree);
        case RBF:
            {
                double sum = 0;
                while(x->index != -1 && y->index !=-1) {
                    if(x->index == y->index) {
                        double d = x->value - y->value;
                        sum += d * d;
                        x++;
                        y++;
                    } else {
                        if(x->index > y->index) {
                            sum += y->value * y->value;
                            y++;
                        } else {
                            sum += x->value * x->value;
                            x++;
                        }
                    }
                }

                while(x->index != -1) {
                    sum += x->value * x->value;
                    x++;
                }

                while(y->index != -1) {
                    sum += y->value * y->value;
                    y++;
                }

                return exp(-param->gamma * sum);
            }
        case SIGMOID:
            return tanh(param->gamma * dot(x, y) + param->coef0);
        case PRECOMPUTED:  //x: test (validation), y: SV
            return x[(int)(y->value)].value;
        default:
            return 0;  // Unreachable
    }
}

#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
    int i, j;
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR) {
        double *sv_coef = model->sv_coef[0];
        double sum = 0;
        for(i = 0; i < model->l; i++) {
            sum += sv_coef[i] * k_function(x, model->SV[i], &model->param);
        }
        sum -= model->rho[0];
        *dec_values = sum;

        if(model->param.svm_type == ONE_CLASS) {
            return (sum > 0) ? 1 : -1;
        } else {
            return sum;
        }
    } else {
        int nr_class = model->nr_class;
        int l = model->l;
        int *start;
        int *vote;
        int p;
        int vote_max_idx;
        double *kvalue = Malloc(double,l);
        for(i = 0; i < l; i++) {
            kvalue[i] = k_function(x, model->SV[i], &model->param);
        }

        start = Malloc(int,nr_class);
        start[0] = 0;
        for(i = 1; i < nr_class; i++) {
            start[i] = start[i - 1] + model->nSV[i - 1];
        }

        vote = Malloc(int,nr_class);
        for(i = 0; i < nr_class; i++) {
            vote[i] = 0;
        }

        p=0;
        for(i = 0; i < nr_class; i++) {
            for(j = i + 1; j < nr_class; j++) {
                double sum = 0;
                int si = start[i];
                int sj = start[j];
                int ci = model->nSV[i];
                int cj = model->nSV[j];

                int k;
                double *coef1 = model->sv_coef[j - 1];
                double *coef2 = model->sv_coef[i];
                for(k = 0; k < ci; k++) {
                    sum += coef1[si + k] * kvalue[si + k];
                }
                for(k = 0; k < cj; k++) {
                    sum += coef2[sj + k] * kvalue[sj + k];
                }
                sum -= model->rho[p];
                dec_values[p] = sum;

                if(dec_values[p] > 0) {
                    vote[i]++;
                } else {
                    vote[j]++;
                }
                p++;
            }
        }

        vote_max_idx = 0;
        for(i = 1; i < nr_class; i++) {
            if(vote[i] > vote[vote_max_idx]) {
                vote_max_idx = i;
            }
        }

        av_free(kvalue);
        av_free(start);
        av_free(vote);
        return model->label[vote_max_idx];
    }
}

static double svm_predict(const svm_model *model, const svm_node *x)
{
    int nr_class = model->nr_class;
    double *dec_values;
    double pred_result;
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR) {
        dec_values = Malloc(double, 1);
    } else {
        dec_values = Malloc(double, nr_class * (nr_class - 1) / 2);
    }
    pred_result = svm_predict_values(model, x, dec_values);
    av_free(dec_values);
    return pred_result;
}

static const char *svm_type_table[] =
{
    "c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[]=
{
    "linear","polynomial","rbf","sigmoid","precomputed",NULL
};

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
    int len;

    if(fgets(line,max_line_len,input) == NULL) {
        return NULL;
    }

    while(strrchr(line, '\n') == NULL) {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL) {
            break;
        }
    }
    return line;
}

/** FSCANF helps to handle fscanf failures.
 * Its do-while block avoids the ambiguity when
 * if (...)
 *    FSCANF();
 * is used
 */
#define FSCANF(_stream, _format, _var) do{ if (fscanf(_stream, _format, _var) != 1) return 0; }while(0)
static int read_model_header(FILE *fp, svm_model* model, AVFilterContext *ctx)
{
    svm_parameter* param = &model->param;
    char cmd[81];
    int i;
    while(1) {
        FSCANF(fp, "%80s", cmd);

        if(av_strcasecmp(cmd, "svm_type") == 0) {
            FSCANF(fp, "%80s", cmd);
            for(i = 0; svm_type_table[i]; i++) {
                if(av_strcasecmp(svm_type_table[i], cmd) == 0) {
                    param->svm_type = i;
                    break;
                }
            }
            if(svm_type_table[i] == NULL) {
                av_log(ctx, AV_LOG_ERROR, "unknown svm type.\n");
                return 0;
            }
        } else if(av_strcasecmp(cmd, "kernel_type") == 0) {
            FSCANF(fp, "%80s", cmd);
            for(i = 0; kernel_type_table[i]; i++) {
                if(av_strcasecmp(kernel_type_table[i], cmd) == 0) {
                    param->kernel_type = i;
                    break;
                }
            }
            if(kernel_type_table[i] == NULL) {
                av_log(ctx, AV_LOG_ERROR, "unknown kernel function.\n");            
                return 0;
            }
        } else if(av_strcasecmp(cmd, "degree") == 0) {
            FSCANF(fp, "%d", &param->degree);
        } else if(av_strcasecmp(cmd, "gamma") == 0) {
            FSCANF(fp, "%lf", &param->gamma);
        } else if(av_strcasecmp(cmd, "coef0") == 0) {
            FSCANF(fp, "%lf", &param->coef0);
        } else if(av_strcasecmp(cmd, "nr_class") == 0) {
            FSCANF(fp,"%d",&model->nr_class);
        } else if(av_strcasecmp(cmd, "total_sv") == 0) {
            FSCANF(fp, "%d", &model->l);
        } else if(av_strcasecmp(cmd, "rho")==0) {
            int n = model->nr_class * (model->nr_class-1) / 2;
            model->rho = Malloc(double, n);
            for(i = 0; i < n; i++) {
                FSCANF(fp, "%lf", &model->rho[i]);
            }
        } else if(av_strcasecmp(cmd, "label") == 0) {
            int n = model->nr_class;
            model->label = Malloc(int, n);
            for(i = 0; i < n; i++) {
                FSCANF(fp, "%d", &model->label[i]);
            }
        } else if(av_strcasecmp(cmd, "probA") == 0) {
            int n = model->nr_class * (model->nr_class - 1) / 2;
            model->probA = Malloc(double, n);
            for(i = 0;i < n; i++) {
                FSCANF(fp, "%lf", &model->probA[i]);
            }
        } else if(av_strcasecmp(cmd, "probB") == 0) {
            int n = model->nr_class * (model->nr_class - 1) / 2;
            model->probB = Malloc(double,n);
            for(i = 0; i < n; i++) {
                FSCANF(fp, "%lf", &model->probB[i]);
            }
        } else if(av_strcasecmp(cmd, "nr_sv") == 0) {
            int n = model->nr_class;
            model->nSV = Malloc(int, n);
            for(i = 0; i < n; i++) {
                FSCANF(fp, "%d", &model->nSV[i]);
            }
        } else if(av_strcasecmp(cmd, "SV") == 0) {
            while(1) {
                int c = getc(fp);
                if(c == EOF || c == '\n') {
                    break;
                }
            }
            break;
        } else {
            av_log(ctx, AV_LOG_ERROR, "unknown text in model file: [%s]\n", cmd);        
            return 0;
        }
    }

    return 1;

}

static svm_model *svm_load_model(const char *model_file_name, AVFilterContext *ctx)
{
    FILE *fp = fopen(model_file_name, "rb");
    int i, j, k, l, m;
    char *p, *endptr, *idx, *val;
    char *old_locale;
    svm_model *model;

    int elements;
    long pos;
    svm_node *x_space;

    if(fp == NULL) {
        return NULL;
    }

    old_locale = strdup(setlocale(LC_ALL, NULL));
    setlocale(LC_ALL, "C");

    // read parameters

    model = Malloc(svm_model,1);
    model->rho = NULL;
    model->probA = NULL;
    model->probB = NULL;
    model->sv_indices = NULL;
    model->label = NULL;
    model->nSV = NULL;

    // read header
    if (!read_model_header(fp, model, ctx)) {
        av_log(ctx, AV_LOG_ERROR, "ERROR: fscanf failed to read model\n");
        setlocale(LC_ALL, old_locale);
        av_free(old_locale);
        av_free(model->rho);
        av_free(model->label);
        av_free(model->nSV);
        av_free(model);
        return NULL;
    }

    // read sv_coef and SV

    elements = 0;
    pos = ftell(fp);

    max_line_len = 1024;
    line = Malloc(char, max_line_len);

    while(readline(fp) != NULL) {
        p = strtok(line, ":");
        while(1) {
            p = strtok(NULL, ":");
            if(p == NULL) {
                break;
            }
            elements++;
        }
    }
    elements += model->l;

    fseek(fp, pos, SEEK_SET);

    m = model->nr_class - 1;
    l = model->l;
    model->sv_coef = Malloc(double *,m);
    for(i = 0; i < m; i++) {
        model->sv_coef[i] = Malloc(double,l);
    }
    model->SV = Malloc(svm_node*, l);
    x_space = NULL;
    if(l > 0) {
        x_space = Malloc(svm_node, elements);
    }

    j=0;
    for(i = 0; i < l; i++) {
        readline(fp);
        model->SV[i] = &x_space[j];

        p = strtok(line, " \t");
        model->sv_coef[0][i] = strtod(p, &endptr);
        for(k = 1; k < m; k++) {
            p = strtok(NULL, " \t");
            model->sv_coef[k][i] = strtod(p, &endptr);
        }

        while(1) {
            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");

            if(val == NULL) {
                break;
            }
            x_space[j].index = (int) strtol(idx, &endptr, 10);
            x_space[j].value = strtod(val, &endptr);

            j++;
        }
        x_space[j++].index = -1;
    }
    av_free(line);

    setlocale(LC_ALL, old_locale);
    av_free(old_locale);

    if (ferror(fp) != 0 || fclose(fp) != 0) {
        return NULL;
    }

    model->free_sv = 1;
    return model;
}

static void svm_free_model_content(svm_model* model_ptr)
{
    int i;
    if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL) {
        av_free((void *) (model_ptr->SV[0]));
    }
    if(model_ptr->sv_coef) {
        for(i = 0; i < model_ptr->nr_class - 1; i++) {
            av_free(model_ptr->sv_coef[i]);
        }
    }

    av_free(model_ptr->SV);
    model_ptr->SV = NULL;

    av_free(model_ptr->sv_coef);
    model_ptr->sv_coef = NULL;

    av_free(model_ptr->rho);
    model_ptr->rho = NULL;

    av_free(model_ptr->label);
    model_ptr->label= NULL;

    av_free(model_ptr->probA);
    model_ptr->probA = NULL;

    av_free(model_ptr->probB);
    model_ptr->probB= NULL;

    av_free(model_ptr->sv_indices);
    model_ptr->sv_indices = NULL;

    av_free(model_ptr->nSV);
    model_ptr->nSV = NULL;
}

static void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
{
    if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL) {
        svm_free_model_content(*model_ptr_ptr);
        av_free(*model_ptr_ptr);
        *model_ptr_ptr = NULL;
    }
}

static void mean(double *score, double curr)
{
    *score += curr;
}

static void min(double *score, double curr)
{
    *score = FFMIN(*score, curr);
}

static void harmonic_mean(double *score, double curr)
{
    *score += 1.0 / (curr + 1.0);
}

#define offset_fn(type, bits) \
    static void offset_##bits##bit(VMAFContext *s, const AVFrame *ref, AVFrame *main, int stride) \
{ \
    int w = s->width; \
    int h = s->height; \
    int i,j; \
    \
    int ref_stride = ref->linesize[0]; \
    int main_stride = main->linesize[0]; \
    \
    const type *ref_ptr = (const type *) ref->data[0]; \
    const type *main_ptr = (const type *) main->data[0]; \
    \
    float *ref_ptr_data = s->ref_data; \
    float *main_ptr_data = s->main_data; \
    \
    for(i = 0; i < h; i++) { \
        for(j = 0; j < w; j++) { \
            ref_ptr_data[j] = (float) ref_ptr[j]; \
            main_ptr_data[j] = (float) main_ptr[j]; \
        } \
        ref_ptr += ref_stride / sizeof(type); \
        ref_ptr_data += stride / sizeof(float); \
        main_ptr += main_stride / sizeof(type); \
        main_ptr_data += stride / sizeof(float); \
    } \
}

offset_fn(uint8_t, 8);
offset_fn(uint16_t, 10);

static int compute_vmaf(const AVFrame *ref, AVFrame *main, void *ctx)
{
    VMAFContext *s = (VMAFContext *) ctx;

    size_t data_sz;
    int i,j;
    ptrdiff_t ref_stride;
    ptrdiff_t ref_px_stride;
    ptrdiff_t stride;
    ptrdiff_t motion_stride;
    ptrdiff_t motion_px_stride;
    int w = s->width;
    int h = s->height;

    ref_stride = ref->linesize[0];

    stride = ALIGN_CEIL(w * sizeof(float));
    motion_stride = ALIGN_CEIL(w * sizeof(uint16_t));
    motion_px_stride = motion_stride / sizeof(uint16_t);

    /** Offset ref and main pixel by OPT_RANGE_PIXEL_OFFSET */
    if (s->desc->comp[0].depth <= 8) {
        offset_8bit(s, ref, main, stride);
    } else {
        offset_10bit(s, ref, main, stride);
    }

    stride = ALIGN_CEIL(s->width * sizeof(float));
    data_sz = (size_t)stride * s->height;

    compute_adm2(s->ref_data, s->main_data, w, h, stride, stride, &s->score,
                 &s->score_num, &s->score_den, s->scores, s->adm_data_buf,
                 s->adm_temp_lo, s->adm_temp_hi);

    append_arr(&s->adm_array, (double)(s->score_num / s->score_den ));
    j = 0;
    for(i = 0; j < 4; i += 2) {
        append_arr(&s->adm_scale_array[j], (double)(s->scores[i] / s->scores[i+1]));
        j++;
    }

    if (s->desc->comp[0].depth <= 8) {
        ref_px_stride = ref_stride / sizeof(uint8_t);
        convolution_f32(s->conv_filter, 5, (const uint8_t *) ref->data[0],
                        s->blur_data, s->temp_data, s->width, s->height,
                        ref_px_stride, motion_px_stride, 8);
    } else {
        ref_px_stride = ref_stride / sizeof(uint16_t);
        convolution_f32(s->conv_filter, 5, (const uint16_t *) ref->data[0],
                        s->blur_data, s->temp_data, s->width, s->height,
                        ref_px_stride, motion_px_stride, 10);
    }

    if(!s->nb_frames) {
        s->score = 0.0;
    } else {
        compute_vmafmotion(s->prev_blur_data, s->blur_data, s->width, s->height,
                           motion_stride, motion_stride, &s->score);
    }

    memcpy(s->prev_blur_data, s->blur_data, data_sz);

    append_arr(&s->motion_array, s->score);

    if(s->nb_frames) {
        append_arr(&s->motion2_array, FFMIN(s->prev_motion_score, s->score));
    }

    s->prev_motion_score = s->score;

    compute_vif2(s->ref_data, s->main_data, w, h, stride, stride, &s->score,
                 &s->score_num, &s->score_den, s->scores, s->vif_data_buf,
                 s->vif_temp);

    j = 0;
    for(i = 0; j < 4; i += 2) {
        append_arr(&s->vif_scale_array[j], (double)((s->scores[i]) / (s->scores[i+1])));
        j++;
    }
    append_arr(&s->vif_array, s->score);

    return 0;
}

static AVFrame *do_vmaf(AVFilterContext *ctx, AVFrame *main, const AVFrame *ref)
{
    VMAFContext *s = ctx->priv;

    compute_vmaf(ref, main, s);

    s->nb_frames++;

    return main;
}

static av_cold int init(AVFilterContext *ctx)
{
    VMAFContext *s = ctx->priv;

    if(!s->called) {
        int i;
        for(i = 0; i < 5; i++) {
            s->conv_filter[i] = lrint(FILTER_5[i] * (1 << N));
        }

        init_arr(&s->adm_array, INIT_FRAMES);
        for(i = 0; i < 4; i++) {
            init_arr(&s->adm_scale_array[i], INIT_FRAMES);
        }
        init_arr(&s->motion_array, INIT_FRAMES);
        init_arr(&s->motion2_array, INIT_FRAMES);
        for(i = 0; i < 4; i++) {
            init_arr(&s->vif_scale_array[i], INIT_FRAMES);
        }
        init_arr(&s->vif_array, INIT_FRAMES);
    }

    s->called = 1;
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

    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

static int config_input_ref(AVFilterLink *inlink)
{
    AVFilterContext *ctx  = inlink->dst;
    VMAFContext *s = ctx->priv;
    int stride;
    size_t data_sz;
    int adm_buf_stride;
    size_t adm_buf_sz;
    int vif_buf_stride;
    size_t vif_buf_sz;

    if (ctx->inputs[0]->w != ctx->inputs[1]->w ||
        ctx->inputs[0]->h != ctx->inputs[1]->h) {
        av_log(ctx, AV_LOG_ERROR, "Width and height of input videos must be same.\n");
        return AVERROR(EINVAL);
    }
    if (ctx->inputs[0]->format != ctx->inputs[1]->format) {
        av_log(ctx, AV_LOG_ERROR, "Inputs must be of same pixel format.\n");
        return AVERROR(EINVAL);
    }

    s->desc = av_pix_fmt_desc_get(inlink->format);
    s->width = ctx->inputs[0]->w;
    s->height = ctx->inputs[0]->h;


    stride = ALIGN_CEIL(s->width * sizeof(float));
    data_sz = (size_t)stride * s->height;

    if (!(s->ref_data = av_malloc(data_sz))) {
        return AVERROR(ENOMEM);
    }

    if (!(s->main_data = av_malloc(data_sz))) {
        return AVERROR(ENOMEM);
    }

    adm_buf_stride = ALIGN_CEIL(((s->width + 1) / 2) * sizeof(float));
    adm_buf_sz = (size_t)adm_buf_stride * ((s->height + 1) / 2);

    if (SIZE_MAX / adm_buf_sz < 35) {
        av_log(ctx, AV_LOG_ERROR, "error: SIZE_MAX / buf_sz_one < 35.\n");
        return AVERROR(EINVAL);
    }

    if (!(s->adm_data_buf = av_malloc(adm_buf_sz * 35))) {
        return AVERROR(ENOMEM);
    }

    if (!(s->adm_temp_lo = av_malloc(stride))) {
        return AVERROR(ENOMEM);
    }
    if (!(s->adm_temp_hi = av_malloc(stride))) {
        return AVERROR(ENOMEM);
    }

    stride = ALIGN_CEIL(s->width * sizeof(uint16_t));
    data_sz = (size_t)stride * s->height;

    if (!(s->prev_blur_data = av_mallocz(data_sz))) {
        return AVERROR(ENOMEM);
    }

    if (!(s->blur_data = av_mallocz(data_sz))) {
        return AVERROR(ENOMEM);
    }

    if (!(s->temp_data = av_mallocz(data_sz))) {
        return AVERROR(ENOMEM);
    }

    vif_buf_stride = ALIGN_CEIL(s->width * sizeof(float));
    vif_buf_sz = (size_t)vif_buf_stride * s->height;

    if (SIZE_MAX / data_sz < 15) {
        av_log(ctx, AV_LOG_ERROR, "error: SIZE_MAX / buf_sz < 15\n");
        return AVERROR(EINVAL);
    }

    if (!(s->vif_data_buf = av_malloc(vif_buf_sz * 16))) {
        return AVERROR(ENOMEM);
    }

    if (!(s->vif_temp = av_malloc(s->width * sizeof(float)))) {
        return AVERROR(ENOMEM);
    }

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
    int i, j;

    if (s->nb_frames > 0) {
        svm_model *svm_model_ptr = svm_load_model(s->model_path, ctx);
        svm_node* nodes = (svm_node*) av_malloc(sizeof(svm_node) * (6 + 1));
        nodes[6].index = -1;
        double prediction;
        double score = 0.0;
        void (*pool_method)(double *score, double curr);
        append_arr(&s->motion2_array, s->prev_motion_score);
        if(!av_strcasecmp(s->pool, "mean")) {
            pool_method = mean;
        } else if(!av_strcasecmp(s->pool, "min")) {
            score = INT_MAX;
            pool_method = min;
        } else if(!av_strcasecmp(s->pool, "harmonic")) {
            pool_method = harmonic_mean;
        }

        for (i = 0; i < s->nb_frames; i++) {
            if (!av_strcasecmp(norm_type, "linear_rescale")) {
                for (j = 0; j < 6; j++) {
                    nodes[j].index = j + 1;
                    if (!av_strcasecmp(feature_names[j], "VMAF_feature_adm2_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->adm_array, i) + (double)(intercepts[j + 1]);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_adm_scale0_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->adm_scale_array[0], i) + (double)(intercepts[j + 1]);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_adm_scale1_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->adm_scale_array[1], i) + (double)(intercepts[j + 1]);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_adm_scale2_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->adm_scale_array[2], i) + (double)(intercepts[j + 1]);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_adm_scale3_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->adm_scale_array[3], i) + (double)(intercepts[j + 1]);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_motion_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->motion_array, i) + (double)(intercepts[j + 1]);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_vif_scale0_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->vif_scale_array[0], i) + (double)(intercepts[j + 1]);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_vif_scale1_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->vif_scale_array[1], i) + (double)(intercepts[j + 1]);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_vif_scale2_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->vif_scale_array[2], i) + (double)(intercepts[j + 1]);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_vif_scale3_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->vif_scale_array[3], i) + (double)(intercepts[j + 1]);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_vif_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->vif_array, i) + (double)(intercepts[j + 1]);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_motion2_score")) {
                        nodes[j].value = (double)(slopes[j + 1]) * get_at_pos(&s->motion2_array, i) + (double)(intercepts[j + 1]);
                    } else {
                        av_log(ctx, AV_LOG_ERROR, "Unknown feature name: %s.\n", feature_names[j]);
                    }
                }
            }
            else {
                for (j = 0; j < 6; j++) {
                    nodes[j].index = j + 1;
                    if (!av_strcasecmp(feature_names[j], "VMAF_feature_adm2_score")) {
                        nodes[j].value = get_at_pos(&s->adm_array, i);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_adm_scale0_score")) {
                        nodes[j].value = get_at_pos(&s->adm_scale_array[0], i);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_adm_scale1_score")) {
                        nodes[j].value = get_at_pos(&s->adm_scale_array[1], i);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_adm_scale2_score")) {
                        nodes[j].value = get_at_pos(&s->adm_scale_array[2], i);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_adm_scale3_score")) {
                        nodes[j].value = get_at_pos(&s->adm_scale_array[3], i);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_motion_score")) {
                        nodes[j].value = get_at_pos(&s->motion_array, i);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_vif_scale0_score")) {
                        nodes[j].value = get_at_pos(&s->vif_scale_array[0], i);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_vif_scale1_score")) {
                        nodes[j].value = get_at_pos(&s->vif_scale_array[1], i);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_vif_scale2_score")) {
                        nodes[j].value = get_at_pos(&s->vif_scale_array[2], i);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_vif_scale3_score")) {
                        nodes[j].value = get_at_pos(&s->vif_scale_array[3], i);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_vif_score")) {
                        nodes[j].value = get_at_pos(&s->vif_array, i);
                    } else if (!av_strcasecmp(feature_names[j], "VMAF_feature_motion2_score")) {
                        nodes[j].value = get_at_pos(&s->motion2_array, i);
                    } else {
                        av_log(ctx, AV_LOG_ERROR, "Unknown feature name: %s.\n", feature_names[j]);
                    }
                }
            }

            prediction = svm_predict(svm_model_ptr, nodes);

            if (!av_strcasecmp(norm_type, "linear_rescale")) {
                /** denormalize */
                prediction = (prediction - (double)(intercepts[0])) / (double)(slopes[0]);
            }
            
            /* score transform */
            if (s->enable_transform) {
                double value = 0.0;

                /* quadratic transform */
                value += (double)(score_transform[0]);
                value += (double)(score_transform[1]) * prediction;
                value += (double)(score_transform[2]) * prediction * prediction;

                /* rectification */
                if (value < prediction) {
                    value = prediction;
                }

                prediction = value;
            }

                pool_method(&score, prediction);
            }

        if(!av_strcasecmp(s->pool, "mean")) {
            s->vmaf_score = score / s->nb_frames;
        } else if(!av_strcasecmp(s->pool, "min")) {
            s->vmaf_score = score;
        } else if(!av_strcasecmp(s->pool, "harmonic")) {
            s->vmaf_score = 1.0 / (score / s->nb_frames) - 1.0;
        }

        av_log(ctx, AV_LOG_INFO, "VMAF Score: %.3f\n", s->vmaf_score);

        svm_free_and_destroy_model((svm_model **)&svm_model_ptr);

        free_arr(&s->adm_array);
        for(i = 0; i < 4; i++) {
            free_arr(&s->adm_scale_array[i]);
        }
        free_arr(&s->motion_array);
        free_arr(&s->motion2_array);
        for(i = 0; i < 4; i++) {
            free_arr(&s->vif_scale_array[i]);
        }
        free_arr(&s->vif_array);

        av_free(s->ref_data);
        av_free(s->main_data);
        av_free(s->adm_data_buf);
        av_free(s->adm_temp_lo);
        av_free(s->adm_temp_hi);
        av_free(s->prev_blur_data);
        av_free(s->blur_data);
        av_free(s->temp_data);
        av_free(s->vif_data_buf);
        av_free(s->vif_temp);
    }

    ff_dualinput_uninit(&s->dinput);
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
