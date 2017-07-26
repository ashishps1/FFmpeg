extern int libsvm_version;

typedef struct
{
    int index;
    double value;
} svm_node;

typedef struct
{
    int l;
    double *y;
    svm_node **x;
} svm_problem;

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };    /** svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /** kernel_type */

typedef struct
{
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
typedef struct
{
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

svm_model *svm_load_model(const char *model_file_name);

double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values);
double svm_predict(const svm_model *model, const svm_node *x);

void svm_free_model_content(svm_model *model_ptr);
void svm_free_and_destroy_model(svm_model** model_ptr_ptr);
