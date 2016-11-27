#ifndef LAPACK_H_
#define LAPACK_H_


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus


void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *A, int *lda,
             double *S, double *U, int *ldu, double *VT, int * ldvt,
             double *work, int *ldwork, int *info);


#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus


#endif  // LAPACK_H_

