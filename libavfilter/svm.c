#include <math.h>
#include <locale.h>
#include "svm.h"
#include "libavutil/opt.h"

#define swap(type, x, y) { type t=x; x=y; y=t; }

static inline double power(double base, int times)
{
    double tmp = base, ret = 1.0;

    for(int t = times; t > 0; t /= 2) {
        if(t % 2 == 1) {
            ret *= tmp;
        };
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
    while(px->index != -1 && py->index != -1)
    {
        if(px->index == py->index)
        {
            sum += px->value * py->value;
            ++px;
            ++py;
        }
        else
        {
            if(px->index > py->index)
                ++py;
            else
                ++px;
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
                while(x->index != -1 && y->index !=-1)
                {
                    if(x->index == y->index)
                    {
                        double d = x->value - y->value;
                        sum += d*d;
                        ++x;
                        ++y;
                    }
                    else
                    {
                        if(x->index > y->index)
                        {
                            sum += y->value * y->value;
                            ++y;
                        }
                        else
                        {
                            sum += x->value * x->value;
                            ++x;
                        }
                    }
                }

                while(x->index != -1)
                {
                    sum += x->value * x->value;
                    ++x;
                }

                while(y->index != -1)
                {
                    sum += y->value * y->value;
                    ++y;
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

double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
    int i;
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
    {
        double *sv_coef = model->sv_coef[0];
        double sum = 0;
        for(i=0;i<model->l;i++)
            sum += sv_coef[i] * k_function(x, model->SV[i], &model->param);
        sum -= model->rho[0];
        *dec_values = sum;

        if(model->param.svm_type == ONE_CLASS)
            return (sum>0)?1:-1;
        else
            return sum;
    }
    else
    {
        int nr_class = model->nr_class;
        int l = model->l;
        int *start;
        int *vote;
        int p;
        int vote_max_idx;
        double *kvalue = Malloc(double,l);
        for(i=0;i<l;i++)
            kvalue[i] = k_function(x, model->SV[i], &model->param);

        start = Malloc(int,nr_class);
        start[0] = 0;
        for(i=1;i<nr_class;i++)
            start[i] = start[i-1]+model->nSV[i-1];

        vote = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
            vote[i] = 0;

        p=0;
        for(i=0;i<nr_class;i++)
            for(int j=i+1;j<nr_class;j++)
            {
                double sum = 0;
                int si = start[i];
                int sj = start[j];
                int ci = model->nSV[i];
                int cj = model->nSV[j];

                int k;
                double *coef1 = model->sv_coef[j-1];
                double *coef2 = model->sv_coef[i];
                for(k=0;k<ci;k++)
                    sum += coef1[si+k] * kvalue[si+k];
                for(k=0;k<cj;k++)
                    sum += coef2[sj+k] * kvalue[sj+k];
                sum -= model->rho[p];
                dec_values[p] = sum;

                if(dec_values[p] > 0)
                    ++vote[i];
                else
                    ++vote[j];
                p++;
            }

        vote_max_idx = 0;
        for(i=1;i<nr_class;i++)
            if(vote[i] > vote[vote_max_idx])
                vote_max_idx = i;

        free(kvalue);
        free(start);
        free(vote);
        return model->label[vote_max_idx];
    }
}

double svm_predict(const svm_model *model, const svm_node *x)
{
    int nr_class = model->nr_class;
    double *dec_values;
    double pred_result;
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
        dec_values = Malloc(double, 1);
    else
        dec_values = Malloc(double, nr_class*(nr_class-1)/2);
    pred_result = svm_predict_values(model, x, dec_values);
    free(dec_values);
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

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var) do{ if (fscanf(_stream, _format, _var) != 1) return 0; }while(0)
static int read_model_header(FILE *fp, svm_model* model)
{
    svm_parameter* param = &model->param;
    char cmd[81];
    while(1)
    {
        FSCANF(fp,"%80s",cmd);

        if(strcmp(cmd,"svm_type")==0)
        {
            int i;
            FSCANF(fp,"%80s",cmd);
            for(i=0;svm_type_table[i];i++)
            {
                if(strcmp(svm_type_table[i],cmd)==0)
                {
                    param->svm_type=i;
                    break;
                }
            }
            if(svm_type_table[i] == NULL)
            {
                fprintf(stderr,"unknown svm type.\n");
                return 0;
            }
        }
        else if(strcmp(cmd,"kernel_type")==0)
        {
            int i;
            FSCANF(fp,"%80s",cmd);
            for(i=0;kernel_type_table[i];i++)
            {
                if(strcmp(kernel_type_table[i],cmd)==0)
                {
                    param->kernel_type=i;
                    break;
                }
            }
            if(kernel_type_table[i] == NULL)
            {
                fprintf(stderr,"unknown kernel function.\n");
                return 0;
            }
        }
        else if(strcmp(cmd,"degree")==0)
            FSCANF(fp,"%d",&param->degree);
        else if(strcmp(cmd,"gamma")==0)
            FSCANF(fp,"%lf",&param->gamma);
        else if(strcmp(cmd,"coef0")==0)
            FSCANF(fp,"%lf",&param->coef0);
        else if(strcmp(cmd,"nr_class")==0)
            FSCANF(fp,"%d",&model->nr_class);
        else if(strcmp(cmd,"total_sv")==0)
            FSCANF(fp,"%d",&model->l);
        else if(strcmp(cmd,"rho")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->rho = Malloc(double,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%lf",&model->rho[i]);
        }
        else if(strcmp(cmd,"label")==0)
        {
            int n = model->nr_class;
            model->label = Malloc(int,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%d",&model->label[i]);
        }
        else if(strcmp(cmd,"probA")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->probA = Malloc(double,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%lf",&model->probA[i]);
        }
        else if(strcmp(cmd,"probB")==0)
        {
            int n = model->nr_class * (model->nr_class-1)/2;
            model->probB = Malloc(double,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%lf",&model->probB[i]);
        }
        else if(strcmp(cmd,"nr_sv")==0)
        {
            int n = model->nr_class;
            model->nSV = Malloc(int,n);
            for(int i=0;i<n;i++)
                FSCANF(fp,"%d",&model->nSV[i]);
        }
        else if(strcmp(cmd,"SV")==0)
        {
            while(1)
            {
                int c = getc(fp);
                if(c==EOF || c=='\n') break;
            }
            break;
        }
        else
        {
            fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
            return 0;
        }
    }

    return 1;

}

svm_model *svm_load_model(const char *model_file_name)
{
    FILE *fp = fopen(model_file_name,"rb");
    int i, j, l, m;
    char *p,*endptr,*idx,*val;
    char *old_locale;
    svm_model *model;
    
    int elements;
    long pos;
    svm_node *x_space;
    
    if(fp==NULL) return NULL;

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
    if (!read_model_header(fp, model))
    {
        fprintf(stderr, "ERROR: fscanf failed to read model\n");
        setlocale(LC_ALL, old_locale);
        free(old_locale);
        free(model->rho);
        free(model->label);
        free(model->nSV);
        free(model);
        return NULL;
    }
    
    // read sv_coef and SV

    elements = 0;
    pos = ftell(fp);

    max_line_len = 1024;
    line = Malloc(char,max_line_len);

    while(readline(fp)!=NULL)
    {
        p = strtok(line,":");
        while(1)
        {
            p = strtok(NULL,":");
            if(p == NULL)
                break;
            ++elements;
        }
    }
    elements += model->l;

    fseek(fp,pos,SEEK_SET);

    m = model->nr_class - 1;
    l = model->l;
    model->sv_coef = Malloc(double *,m);
    for(i=0;i<m;i++)
        model->sv_coef[i] = Malloc(double,l);
    model->SV = Malloc(svm_node*,l);
    x_space = NULL;
    if(l>0) x_space = Malloc(svm_node,elements);

    j=0;
    for(i=0;i<l;i++)
    {
        readline(fp);
        model->SV[i] = &x_space[j];

        p = strtok(line, " \t");
        model->sv_coef[0][i] = strtod(p,&endptr);
        for(int k=1;k<m;k++)
        {
            p = strtok(NULL, " \t");
            model->sv_coef[k][i] = strtod(p,&endptr);
        }

        while(1)
        {
            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");

            if(val == NULL)
                break;
            x_space[j].index = (int) strtol(idx,&endptr,10);
            x_space[j].value = strtod(val,&endptr);

            ++j;
        }
        x_space[j++].index = -1;
    }
    free(line);

    setlocale(LC_ALL, old_locale);
    free(old_locale);

    if (ferror(fp) != 0 || fclose(fp) != 0)
        return NULL;

    model->free_sv = 1;
    return model;
}

void svm_free_model_content(svm_model* model_ptr)
{
    if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
        free((void *)(model_ptr->SV[0]));
    if(model_ptr->sv_coef)
    {
        for(int i=0;i<model_ptr->nr_class-1;i++)
            free(model_ptr->sv_coef[i]);
    }

    free(model_ptr->SV);
    model_ptr->SV = NULL;

    free(model_ptr->sv_coef);
    model_ptr->sv_coef = NULL;

    free(model_ptr->rho);
    model_ptr->rho = NULL;

    free(model_ptr->label);
    model_ptr->label= NULL;

    free(model_ptr->probA);
    model_ptr->probA = NULL;

    free(model_ptr->probB);
    model_ptr->probB= NULL;

    free(model_ptr->sv_indices);
    model_ptr->sv_indices = NULL;

    free(model_ptr->nSV);
    model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(svm_model** model_ptr_ptr)
{
    if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
    {
        svm_free_model_content(*model_ptr_ptr);
        free(*model_ptr_ptr);
        *model_ptr_ptr = NULL;
    }
}
