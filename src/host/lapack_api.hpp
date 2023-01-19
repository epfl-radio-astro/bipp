#pragma once

#include <complex>
#include <vector>

#include "bipp/bipp.h"
#include "bipp/config.h"
extern "C" {

enum class LapackeLayout { ROW_MAJOR = 101, COL_MAJOR = 102 };

#ifdef BIPP_LAPACK_C
float LAPACKE_slamch(char cmach);
double LAPACKE_dlamch(char cmach);

int LAPACKE_cheevx(int matrix_layout, char jobz, char range, char uplo, int n, void* a, int lda,
                   float vl, float vu, int il, int iu, float abstol, int* m, float* w, void* z,
                   int ldz, int* ifail);
int LAPACKE_zheevx(int matrix_layout, char jobz, char range, char uplo, int n, void* a, int lda,
                   double vl, double vu, int il, int iu, double abstol, int* m, double* w, void* z,
                   int ldz, int* ifail);
int LAPACKE_chegvx(int matrix_layout, int itype, char jobz, char range, char uplo, int n, void* a,
                   int lda, void* b, int ldb, float vl, float vu, int il, int iu, float abstol,
                   int* m, float* w, void* z, int ldz, int* ifail);
int LAPACKE_zhegvx(int matrix_layout, int itype, char jobz, char range, char uplo, int n, void* a,
                   int lda, void* b, int ldb, double vl, double vu, int il, int iu, double abstol,
                   int* m, double* w, void* z, int ldz, int* ifail);
#else
float slamch_(const char* cmach, int cmach_len);

double dlamch_(const char* cmach, int cmach_len);

void cheevx_(const char* JOBZ, const char* RANGE, const char* UPLO, const int* N, void* A,
             const int* LDA, const void* VL, const void* VU, const int* IL, const int* IU,
             const void* ABSTOL, const int* M, void* W, void* Z, const int* LDZ, void* WORK,
             int* LWORK, void* RWORK, int* IWORK, int* IFAIL, int* INFO, int JOBZ_len,
             int RANGE_len, int UPLO_len);

void zheevx_(const char* JOBZ, const char* RANGE, const char* UPLO, const int* N, void* A,
             const int* LDA, const void* VL, const void* VU, const int* IL, const int* IU,
             const void* ABSTOL, const int* M, void* W, void* Z, const int* LDZ, void* WORK,
             int* LWORK, void* RWORK, int* IWORK, int* IFAIL, int* INFO, int JOBZ_len,
             int RANGE_len, int UPLO_len);

void chegvx_(const int* ITYPE, const char* JOBZ, const char* RANGE, const char* UPLO, const int* N,
             void* A, const int* LDA, void* B, const int* LDB, const void* VL, const void* VU,
             const int* IL, const int* IU, const void* ABSTOL, const int* M, void* W, void* Z,
             const int* LDZ, void* WORK, int* LWORK, void* RWORK, int* IWORK, int* IFAIL, int* INFO,
             int JOBZ_len, int RANGE_len, int UPLO_len);

void zhegvx_(const int* ITYPE, const char* JOBZ, const char* RANGE, const char* UPLO, const int* N,
             void* A, const int* LDA, void* B, const int* LDB, const void* VL, const void* VU,
             const int* IL, const int* IU, const void* ABSTOL, const int* M, void* W, void* Z,
             const int* LDZ, void* WORK, int* LWORK, void* RWORK, int* IWORK, int* IFAIL, int* INFO,
             int JOBZ_len, int RANGE_len, int UPLO_len);

#endif
}

namespace bipp {
namespace host {
namespace lapack {

inline auto slamch(char cmach) -> float {
#ifdef BIPP_LAPACK_C
  return LAPACKE_slamch(cmach);
#else
  return slamch_(&cmach, 1);
#endif
}

inline auto dlamch(char cmach) -> double {
#ifdef BIPP_LAPACK_C
  return LAPACKE_dlamch(cmach);
#else
  return dlamch_(&cmach, 1);
#endif
}

inline auto eigh_solve(LapackeLayout matrixLayout, char jobz, char range, char uplo, int n,
                       std::complex<float>* a, int lda, float vl, float vu, int il, int iu, int* m,
                       float* w, std::complex<float>* z, int ldz, int* ifail) -> int {
#ifdef BIPP_LAPACK_C
  return LAPACKE_cheevx(static_cast<int>(matrixLayout), jobz, range, uplo, n, a, lda, vl, vu, il,
                        iu, 2 * LAPACKE_slamch('S'), m, w, z, ldz, ifail);
#else
  int info = 0;
  float abstol = 2 * slamch_("S", 1);
  int lwork = -1;
  std::vector<float> rwork(7 * n);
  std::vector<int> iwork(5 * n);
  std::complex<float> worksize = 0;

  // get work buffer size
  cheevx_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, &worksize,
          &lwork, rwork.data(), iwork.data(), ifail, &info, 1, 1, 1);
  lwork = static_cast<int>(worksize.real());
  if (lwork < 2 * n) lwork = 2 * n;

  std::vector<std::complex<float> > work(lwork);
  cheevx_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz,
          work.data(), &lwork, rwork.data(), iwork.data(), ifail, &info, 1, 1, 1);

  return info;
#endif
}

inline auto eigh_solve(LapackeLayout matrixLayout, char jobz, char range, char uplo, int n,
                       std::complex<double>* a, int lda, double vl, double vu, int il, int iu,
                       int* m, double* w, std::complex<double>* z, int ldz, int* ifail) -> int {
#ifdef BIPP_LAPACK_C
  return LAPACKE_zheevx(static_cast<int>(matrixLayout), jobz, range, uplo, n, a, lda, vl, vu, il,
                        iu, 2 * LAPACKE_dlamch('S'), m, w, z, ldz, ifail);
#else
  int info = 0;
  double abstol = 2 * dlamch_("S", 1);
  int lwork = -1;
  std::vector<double> rwork(7 * n);
  std::vector<int> iwork(5 * n);
  std::complex<double> worksize = 0;

  // get work buffer size
  zheevx_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz, &worksize,
          &lwork, rwork.data(), iwork.data(), ifail, &info, 1, 1, 1);
  lwork = static_cast<int>(worksize.real());
  if (lwork < 2 * n) lwork = 2 * n;

  std::vector<std::complex<double> > work(lwork);
  zheevx_(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, m, w, z, &ldz,
          work.data(), &lwork, rwork.data(), iwork.data(), ifail, &info, 1, 1, 1);

  return info;
#endif
}

inline auto eigh_solve(LapackeLayout matrixLayout, int itype, char jobz, char range, char uplo,
                       int n, std::complex<float>* a, int lda, std::complex<float>* b, int ldb,
                       float vl, float vu, int il, int iu, int* m, float* w, std::complex<float>* z,
                       int ldz, int* ifail) -> int {
#ifdef BIPP_LAPACK_C
  return LAPACKE_chegvx(static_cast<int>(matrixLayout), itype, jobz, range, uplo, n, a, lda, b, ldb,
                        vl, vu, il, iu, 2 * LAPACKE_slamch('S'), m, w, z, ldz, ifail);
#else
  int info = 0;
  float abstol = 2 * slamch_("S", 1);
  int lwork = -1;
  std::vector<float> rwork(7 * n);
  std::vector<int> iwork(5 * n);
  std::complex<float> worksize = 0;

  // get work buffer size
  chegvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, m, w, z,
          &ldz, &worksize, &lwork, rwork.data(), iwork.data(), ifail, &info, 1, 1, 1);
  lwork = static_cast<int>(worksize.real());
  if (lwork < 2 * n) lwork = 2 * n;

  std::vector<std::complex<float> > work(lwork);
  chegvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, m, w, z,
          &ldz, work.data(), &lwork, rwork.data(), iwork.data(), ifail, &info, 1, 1, 1);

  return info;
#endif
}

inline auto eigh_solve(LapackeLayout matrixLayout, int itype, char jobz, char range, char uplo,
                       int n, std::complex<double>* a, int lda, std::complex<double>* b, int ldb,
                       double vl, double vu, int il, int iu, int* m, double* w,
                       std::complex<double>* z, int ldz, int* ifail) -> int {
#ifdef BIPP_LAPACK_C
  return LAPACKE_zhegvx(static_cast<int>(matrixLayout), itype, jobz, range, uplo, n, a, lda, b, ldb,
                        vl, vu, il, iu, 2 * LAPACKE_dlamch('S'), m, w, z, ldz, ifail);
#else
  int info = 0;
  double abstol = 2 * dlamch_("S", 1);
  int lwork = -1;
  std::vector<double> rwork(7 * n);
  std::vector<int> iwork(5 * n);
  std::complex<double> worksize = 0;

  // get work buffer size
  zhegvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, m, w, z,
          &ldz, &worksize, &lwork, rwork.data(), iwork.data(), ifail, &info, 1, 1, 1);
  lwork = static_cast<int>(worksize.real());
  if (lwork < 2 * n) lwork = 2 * n;

  std::vector<std::complex<double> > work(lwork);
  zhegvx_(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il, &iu, &abstol, m, w, z,
          &ldz, work.data(), &lwork, rwork.data(), iwork.data(), ifail, &info, 1, 1, 1);

  return info;
#endif
}
}  // namespace lapack
}  // namespace host
}  // namespace bipp
