// y[0] == T
// y[1] == T*
// y[2] == CTL
// y[3] == V
// 0 == s1
// 1 == s3
// 2 == r
// 3 == k1
// 4 == μ_T
// 5 == μ_T*
// 6 == T_max
// 7 == p_4
// 8 == k_8
// 9 == p_8
// 10 == μ_ctl
// 11 == μ_V
// 12 == p_V
// 13 == k_2
#include"gsl_odeiv2.h"
#include"gsl_multifit_nlinear.h"
#include"gsl_matrix.h"
#include"gsl_vector.h"
#include"gsl_blas.h"
#include<stdio.h>

int N=8;

double val[32];

struct data{
    size_t n;
    double *t;
    double *y;
};

int rhsF(double t, const double y[], double f[], void *params){
    double *buf = (double*)params;
    double k = 1 - (y[0]+y[1])/buf[6];
    f[0] = buf[0] + buf[2]*y[0]*k - buf[3]*y[0]*y[3] - buf[4]*y[0];
    f[1] = buf[3]*y[0]*y[3] + buf[7]*y[1]*k - buf[8]*y[1]*y[2] - buf[5]*y[1];
    f[2] = buf[1] + buf[9]*y[1]*y[2] - buf[10]*y[2];
    f[3] = buf[12]*y[1] - buf[13]*y[0]*y[3] - buf[11]*y[3];
    if(f[3]<0) f[3] = 0;
    return  GSL_SUCCESS;
}

int jacF(double t, const double y[], double *dfdy, double dfdt[], void *params){
    (void) (t);
    double *buf = (double*)params;
    gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, 4, 4);
    gsl_matrix *m = &dfdy_mat.matrix;
    gsl_matrix_set(m, 0, 0, buf[2]*(1 - (y[0]+y[1])/buf[6]) - buf[2]*y[0]/buf[6] - buf[3]*y[3] - buf[4]);
    gsl_matrix_set(m, 0, 1, -1.0*buf[2]*y[0]/buf[6]);
    gsl_matrix_set(m, 0, 2, 0.0);
    gsl_matrix_set(m, 0, 3, -1.0*buf[3]*y[0]);
    gsl_matrix_set(m, 1, 0, buf[3]*y[3] - buf[7]*y[1]/buf[6]);
    gsl_matrix_set(m, 1, 1, buf[7]*(buf[6] - y[0] - 2*y[1])/buf[6] - buf[8]*y[2] - buf[5]);
    gsl_matrix_set(m, 1, 2, -1.0*buf[8]*y[1]);
    gsl_matrix_set(m, 1, 3, buf[3]*y[0]);
    gsl_matrix_set(m, 2, 0, 0.0);
    gsl_matrix_set(m, 2, 1, buf[9]*y[2]);
    gsl_matrix_set(m, 2, 2, buf[9]*y[1] - buf[10]);
    gsl_matrix_set(m, 2, 3, 0.0);
    gsl_matrix_set(m, 3, 0, buf[13]*y[3]);
    gsl_matrix_set(m, 3, 1, buf[12]);
    gsl_matrix_set(m, 3, 2, 0.0);
    gsl_matrix_set(m, 3, 3, -1.0*buf[13]*y[0] - buf[11]);
    dfdt[0] = 0.0;
    dfdt[1] = 0.0;
    dfdt[2] = 0.0;
    dfdt[3] = 0.0;
    return GSL_SUCCESS;
}

int expb_df(const gsl_vector *x, void *data, gsl_matrix *J){
    int n = ((struct data*)data)->n;
    double *t = ((struct data*)data)->t;
    double *yi = ((struct data*)data)->y;

    double k1 = gsl_vector_get(x, 0);
    double k8 = gsl_vector_get(x, 1);
    double p8 = gsl_vector_get(x, 2);
    double pv = gsl_vector_get(x, 3);

    double buf[14] = {10.0, 5.0, 0.03883, k1, 0.02, 0.28, 1600.0, 0.002, k8, p8, 0.015, 3.0, pv, 7.79e-6};
    gsl_matrix_set(J, 0, 0, -1.0);
    gsl_matrix_set(J, 0, 1, 0.0);
    gsl_matrix_set(J, 0, 2, 0.0);
    gsl_matrix_set(J, 0, 3, 0.0);

    gsl_matrix_set(J, 1, 0, 1.0);
    gsl_matrix_set(J, 1, 1, -1.0);
    gsl_matrix_set(J, 1, 2, 0.0);
    gsl_matrix_set(J, 1, 3, 0.0);

    gsl_matrix_set(J, 2, 0, 0.0);
    gsl_matrix_set(J, 2, 1, 0.0);
    gsl_matrix_set(J, 2, 2, 1.0);
    gsl_matrix_set(J, 2, 3, 0.0);

    gsl_matrix_set(J, 3, 0, 0.0);
    gsl_matrix_set(J, 3, 1, 0.0);
    gsl_matrix_set(J, 3, 2, 0.0);
    gsl_matrix_set(J, 3, 3, 1.0);
    
    return GSL_SUCCESS;
}

int expb_f(const gsl_vector *x, void *data, gsl_vector *f){
    int k = 0;
    size_t n = ((struct data*)data)->n;
    double *t = ((struct data*)data)->t;
    double *yi = ((struct data*)data)->y;
    double k1 = gsl_vector_get(x, 0);
    double k8 = gsl_vector_get(x, 1);
    double p8 = gsl_vector_get(x, 2);
    double pv = gsl_vector_get(x, 3);
    
    double params[14] = {10.0, 5.0, 0.03883, k1, 0.02, 0.28, 1600.0, 0.002, k8, p8, 0.015, 3.0, pv, 7.79e-6};
    
    const gsl_odeiv2_step_type *T = gsl_odeiv2_step_msbdf;

    gsl_odeiv2_step *s = gsl_odeiv2_step_alloc (T, 4);
    
    gsl_odeiv2_system odeSystem = {rhsF, jacF, 4, &params};

    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&odeSystem, T, 1e-6, 1e-6, 0.0);

    gsl_odeiv2_step_set_driver(s, d);
    
    double t0 = 0.0, t1 = 337.0;
    double y[4] = {1138.0, 0.0, 0.0, 0.0};
    for(int i = 1; i<=337; i++){

        double ti = i*t1/337.0;
        int status = gsl_odeiv2_driver_apply(d, &t0, ti, y);

        if(status != GSL_SUCCESS){
            printf("error, return value = %d\n", status);
            break;
        }
        if(i == t[i]){
            val[4*k] = y[0];
            val[4*k+1] = y[1];
            val[4*k+2] = y[2];
            val[4*k+3] = y[3];
            gsl_vector_set(f, k, (y[0] - yi[i])*(y[0] - yi[i]) + (y[3]- yi[8+i])*(y[3] - yi[8+i]));
            printf("%.5f", (y[0] - yi[i])*(y[0] - yi[i]) + (y[3]- yi[8+i])*(y[3] - yi[8+i]));
            k++;
        }

    }
    gsl_odeiv2_step_free(s);
    gsl_odeiv2_driver_free(d);
    return GSL_SUCCESS;
}

void callback(const size_t iter, void *params, const gsl_multifit_nlinear_workspace *w){
    gsl_vector *f = gsl_multifit_nlinear_residual(w);
    gsl_vector *x = gsl_multifit_nlinear_position(w);
    double rcond;

    gsl_multifit_nlinear_rcond(&rcond, w);

    fprintf(stderr, "iter %2zu: k1 = %.4f, k8 = %.4f, p8 = %.4f, b = %.4f, cond(J) = %8.4f, |f(x)| = %.4f\n", iter, 
            gsl_vector_get(x, 0),
            gsl_vector_get(x, 1),
            gsl_vector_get(x, 2),
            gsl_vector_get(x, 3),
            1.0/rcond,
            gsl_blas_dnrm2(f));
}


int main(){
    double params[14] = { 10.0, 5.0, 0.02, 4.86e-6, 0.02, 0.28, 1500.0, 0.002, 0.0126, 0.00335, 0.015, 3.0, 6600.03, 7.79e-6};
    
    const gsl_odeiv2_step_type *T = gsl_odeiv2_step_msbdf;

    gsl_odeiv2_step *s = gsl_odeiv2_step_alloc (T, 4);
    
    gsl_odeiv2_system odeSystem = {rhsF, jacF, 4, &params};

    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&odeSystem, T, 1e-6, 1e-6, 0.0);

    gsl_odeiv2_step_set_driver(s, d);

    double t[8], y[16];

    t[0] = 0;
    y[0] = 1138;
    y[8] = 0;
    t[1] = 10;
    y[1] = 822.118;
    y[9] = 541;
    t[2] = 17;
    y[2] = 515.74;
    y[10] = 769;
    t[3] = 24;
    y[3] = 515.74;
    y[11] = 640;
    t[4] = 31;
    y[4] = 541.27;
    y[12] = 554;
    t[5] = 38;
    y[5] = 520.82;
    y[13] = 541;
    t[6] = 175;
    y[6] = 571.9;
    y[14] = 452;
    t[7] = 337;
    y[7] = 612.76;
    y[15] = 517;
    int k = 1;
    double t0 = 0.0, t1 = 337.0;   
    double yi[4] = { 1138.0, 0.0, 0.0, 0.0 };
    double sum = 0;
    for(int i = 1; i<=337; i++){
        
        double ti = i*t1/337.0;

        int status = gsl_odeiv2_driver_apply( d, &t0, ti, yi);

        if (status != GSL_SUCCESS){
            printf("error, return value=%d\n", status);
            break;
        }
        if(t[k] == ti){
            printf("t = %.5e  Ti = %.5e Vi = %.5e   T = %.5e V = %.5e\n",t[k], y[k], y[k+8], yi[0], yi[3]);
            sum = sum + (y[k] - yi[0])*(y[k] - yi[0]) + (y[k+8] - yi[3])*(y[k+8] - yi[3]);
            k++;
        }
        //printf("%.5e %.5e %.5e %.5e %.5e\n", t0, y[0], y[1], y[2], y[3]);
    }
    printf("\n sum: %.5e\n", sum);
    gsl_odeiv2_step_free(s);
    gsl_odeiv2_driver_free(d);
/*
    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_workspace *w;
    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
    const size_t n = 8;
    const size_t p = 4;

    gsl_vector *f;
    gsl_matrix *J;
    gsl_matrix *covar = gsl_matrix_alloc (p, p);
    double t[N], y[N*2], weights[N];
    struct data d = {n, t, y};
    double x_init[4] = {4.86e-04, 0.019, 0.008, 6300};
    gsl_vector_view x = gsl_vector_view_array (x_init, p);
    gsl_vector_view wts = gsl_vector_view_array (weights, n);
    double chisq, chisq0;
    int info, status;
    size_t i;

    const double xtol = 1e-8;
    const double gtol = 1e-8;
    const double ftol = 0.0;

    fdf.f = expb_f;
    fdf.df = expb_df;
    fdf.n = n;
    fdf.p = p;
    fdf.params = &d;

    t[0] = 0;
    y[0] = 1138;
    y[8] = 0;
    t[1] = 10;
    y[1] = 822.118;
    y[9] = 541;
    t[2] = 17;
    y[2] = 515.74;
    y[10] = 769;
    t[3] = 24;
    y[3] = 515.74;
    y[11] = 640;
    t[4] = 31;
    y[4] = 541.27;
    y[12] = 554;
    t[5] = 38;
    y[5] = 520.82;
    y[13] = 541;
    t[6] = 175;
    y[6] = 571.9;
    y[14] = 452;
    t[7] = 337;
    y[7] = 612.76;
    y[15] = 517;
    
    w = gsl_multifit_nlinear_alloc (T, &fdf_params, n, p);
    
    gsl_multifit_nlinear_winit (&x.vector, &wts.vector, &fdf, w);
    
    f = gsl_multifit_nlinear_residual(w);
    
    gsl_blas_ddot(f, f, &chisq0);
    
    status = gsl_multifit_nlinear_driver(100, xtol, gtol, ftol, callback, NULL, &info, w);

    fprintf(stderr, "summary from method '%s/%s'\n", gsl_multifit_nlinear_name(w), gsl_multifit_nlinear_trs_name(w));
    fprintf(stderr, "number of iterations: %zu\n", gsl_multifit_nlinear_niter(w));
    fprintf(stderr, "function evaluations: %zu\n", fdf.nevalf);
    fprintf(stderr, "Jacobian evaluations: %zu\n", fdf.nevaldf);
    fprintf(stderr, "reason for stopping: %s\n", (info == 1) ? "small step size" : "small gradient");
    fprintf(stderr, "initial |f(x)| = %f\n", sqrt(chisq0));
    fprintf(stderr, "final |f(x)| = %f\n", sqrt(chisq));*/
    return 0;
}
