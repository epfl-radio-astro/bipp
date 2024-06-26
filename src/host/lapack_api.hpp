#pragma once

#include <complex>
#include <vector>
#include <iostream>
#include <omp.h>


#include "bipp/config.h"
#include "bipp/exceptions.hpp"

extern "C" {

enum class LapackeLayout { ROW_MAJOR = 101, COL_MAJOR = 102 };

#ifdef BIPP_LAPACK_C
float LAPACKE_slamch(char cmach);

double LAPACKE_dlamch(char cmach);

int LAPACKE_chegv(int matrix_layout, int itype, char jobz, char uplo, int n, void* a, int lda,
                  void* b, int ldb, float* w);

int LAPACKE_zhegv(int matrix_layout, int itype, char jobz, char uplo, int n, void* a, int lda,
                  void* b, int ldb, double* w);

int LAPACKE_cheev(int matrix_layout, char jobz, char uplo, int n, void* a, int lda, float* w);

int LAPACKE_zheev(int matrix_layout, char jobz, char uplo, int n, void* a, int lda, double* w);

#else
float slamch_(const char* cmach, int cmach_len);

double dlamch_(const char* cmach, int cmach_len);

void cheev_(char* jobz, char* uplo, int* n, void* a, int* lda, float* w, void* work, int* lwork,
            float* rwork, int* info, int JOBZ_len, int UPLO_len);

void zheev_(char* jobz, char* uplo, int* n, void* a, int* lda, double* w, void* work, int* lwork,
            double* rwork, int* info, int JOBZ_len, int UPLO_len);

void chegv_(int const* itype, char const* jobz, char const* uplo, int const* n, void* A,
            int const* lda, void* B, int const* ldb, float* W, void* work, int const* lwork,
            float* rwork, int* info, int JOBZ_len, int UPLO_len);

void zhegv_(int const* itype, char const* jobz, char const* uplo, int const* n, void* A,
            int const* lda, void* B, int const* ldb, double* W, void* work, int const* lwork,
            double* rwork, int* info, int JOBZ_len, int UPLO_len);

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

inline auto set_omp_num_threads() -> int {
  int env_omp_num_threads = 1;
  if (const char* env_omp_num_threads_ = std::getenv("OMP_NUM_THREADS")) {
      env_omp_num_threads = atoi(env_omp_num_threads_);
  }
  int lapack_nt = 1;
  if (const char* lapack_nt_ = std::getenv("BIPP_LAPACK_C_NUM_THREADS")) {
    const int lnt_tmp = atoi(lapack_nt_);
    if (lnt_tmp > 0) lapack_nt = lnt_tmp;
  }
  //std::cout << "@@@@  lapack_nt " << lapack_nt << ", env_omp_num_threads " << env_omp_num_threads << '\n';
  omp_set_num_threads(lapack_nt);

  return env_omp_num_threads;
}

inline auto eigh_solve(LapackeLayout matrixLayout, char jobz, char uplo, int n,
                       std::complex<float>* a, int lda, float* w) -> void {
#ifdef BIPP_LAPACK_C
  auto env_omp_num_threads = set_omp_num_threads();
  const auto info = LAPACKE_cheev(static_cast<int>(matrixLayout), jobz, uplo, n, a, lda, w);
  omp_set_num_threads(env_omp_num_threads);
#else
  int info = 0;
  int lwork = -1;
  std::vector<float> rwork(3 * n - 2);
  std::complex<float> worksize = 0;

  // get work buffer size
  cheev_(&jobz, &uplo, &n, a, &lda, w, &worksize, &lwork, rwork.data(), &info, 1, 1);
  lwork = static_cast<int>(worksize.real());
  if (lwork < 2 * n) lwork = 2 * n;

  std::vector<std::complex<float> > work(lwork);
  cheev_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, rwork.data(), &info, 1, 1);
#endif
  if (info) throw EigensolverError();
}

inline auto eigh_solve(LapackeLayout matrixLayout, char jobz, char uplo, int n,
                       std::complex<double>* a, int lda, double* w) -> void {
#ifdef BIPP_LAPACK_C
  auto env_omp_num_threads = set_omp_num_threads();
  const auto info = LAPACKE_zheev(static_cast<int>(matrixLayout), jobz, uplo, n, a, lda, w);
  omp_set_num_threads(env_omp_num_threads);
#else
  int info = 0;
  int lwork = -1;
  std::vector<double> rwork(3 * n - 2);
  std::complex<double> worksize = 0;

  // get work buffer size
  zheev_(&jobz, &uplo, &n, a, &lda, w, &worksize, &lwork, rwork.data(), &info, 1, 1);
  lwork = static_cast<int>(worksize.real());
  if (lwork < 2 * n) lwork = 2 * n;

  std::vector<std::complex<double> > work(lwork);
  zheev_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, rwork.data(), &info, 1, 1);
#endif
  if (info) throw EigensolverError();
}

inline auto eigh_solve(LapackeLayout matrixLayout, int itype, char jobz, char uplo, int n,
                       std::complex<float>* a, int lda, std::complex<float>* b, int ldb, float* w)
    -> void {
#ifdef BIPP_LAPACK_C
  auto env_omp_num_threads = set_omp_num_threads();
  const auto info = LAPACKE_chegv(static_cast<int>(matrixLayout), itype, jobz, uplo, n, a, lda, b, ldb, w);
  omp_set_num_threads(env_omp_num_threads);
#else
  int info = 0;
  int lwork = -1;
  std::vector<float> rwork(3 * n - 2);
  std::complex<float> worksize = 0;

  // get work buffer size
  chegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &worksize, &lwork, rwork.data(), &info, 1,
         1);
  lwork = static_cast<int>(worksize.real());
  if (lwork < 2 * n) lwork = 2 * n;

  std::vector<std::complex<float> > work(lwork);
  chegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork, rwork.data(), &info, 1,
         1);
#endif
  if (info) throw EigensolverError();
}

inline auto eigh_solve(LapackeLayout matrixLayout, int itype, char jobz, char uplo, int n,
                       std::complex<double>* a, int lda, std::complex<double>* b, int ldb, double* w)
    -> void {
#ifdef BIPP_LAPACK_C
  auto env_omp_num_threads = set_omp_num_threads();
  const auto info = LAPACKE_zhegv(static_cast<int>(matrixLayout), itype, jobz, uplo, n, a, lda, b, ldb, w);
  omp_set_num_threads(env_omp_num_threads);
#else
  int info = 0;
  int lwork = -1;
  std::vector<double> rwork(3 * n - 2);
  std::complex<double> worksize = 0;

  // get work buffer size
  zhegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &worksize, &lwork, rwork.data(), &info, 1,
         1);
  lwork = static_cast<int>(worksize.real());
  if (lwork < 2 * n) lwork = 2 * n;

  std::vector<std::complex<double> > work(lwork);
  zhegv_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork, rwork.data(), &info, 1,
         1);
#endif
  if (info) throw EigensolverError();
}

}  // namespace lapack
}  // namespace host
}  // namespace bipp
