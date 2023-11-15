#include "gpu/util/solver_api.hpp"

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"
#include "host/lapack_api.hpp"
#include "memory/array.hpp"

#ifdef BIPP_MAGMA
#include <magma.h>
#include <magma_c.h>
#include <magma_types.h>
#include <magma_z.h>
#else
#include <cusolverDn.h>
#endif

namespace bipp {
namespace gpu {
namespace eigensolver {

namespace {
#ifdef BIPP_MAGMA
struct MagmaInit {
  MagmaInit() { magma_init(); }
  MagmaInit(const MagmaInit&) = delete;
  ~MagmaInit() { magma_finalize(); }
};

MagmaInit MAGMA_INIT_GUARD;
#endif

auto convert_jobz(char j) {
#ifdef BIPP_MAGMA
  switch (j) {
    case 'N':
    case 'n':
      return MagmaNoVec;
    case 'V':
    case 'v':
      return MagmaVec;
  }
  throw InternalError();
  return MagmaVec;
#else
  switch (j) {
    case 'N':
    case 'n':
      return CUSOLVER_EIG_MODE_NOVECTOR;
    case 'V':
    case 'v':
      return CUSOLVER_EIG_MODE_VECTOR;
  }
  throw InternalError();
  return CUSOLVER_EIG_MODE_VECTOR;
#endif
}

auto convert_range(char r) {
#ifdef BIPP_MAGMA
  switch (r) {
    case 'A':
    case 'a':
      return MagmaRangeAll;
    case 'V':
    case 'v':
      return MagmaRangeV;
    case 'I':
    case 'i':
      return MagmaRangeI;
  }
  throw InternalError();
  return MagmaRangeAll;
#else
  switch (r) {
    case 'A':
    case 'a':
      return CUSOLVER_EIG_RANGE_ALL;
    case 'V':
    case 'v':
      return CUSOLVER_EIG_RANGE_V;
    case 'I':
    case 'i':
      return CUSOLVER_EIG_RANGE_I;
  }
  throw InternalError();
  return CUSOLVER_EIG_RANGE_ALL;
#endif
}

auto convert_uplo(char u) {
#ifdef BIPP_MAGMA
  switch (u) {
    case 'L':
    case 'l':
      return MagmaLower;
    case 'U':
    case 'u':
      return MagmaUpper;
  }
  throw InternalError();
  return MagmaLower;
#else
  switch (u) {
    case 'L':
    case 'l':
      return CUBLAS_FILL_MODE_LOWER;
    case 'U':
    case 'u':
      return CUBLAS_FILL_MODE_UPPER;
  }
  throw InternalError();
  return CUBLAS_FILL_MODE_LOWER;
#endif
}

}  // namespace

auto solve(Queue& queue, char jobz, char uplo, int n, api::ComplexType<float>* a, int lda, float* w)
    -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto uploEnum = convert_uplo(uplo);

#ifdef BIPP_MAGMA
  auto rangeEnum = convert_range('A');
  queue.sync();
  const float abstol = 2 * host::lapack::slamch('S');

  auto z = queue.create_device_array<api::ComplexType<float>, 1>(n * n);
  int ldz = n;

  auto wHost = queue.create_pinned_array<float, 1>(n);
  auto wA = queue.create_host_array<api::ComplexType<float>, 1>(n * n);
  auto wZ = queue.create_host_array<api::ComplexType<float>, 1>(n * n);
  auto rwork = queue.create_host_array<float, 1>(7 * n);
  auto iwork = queue.create_host_array<int, 1>(5 * n);
  auto ifail = queue.create_host_array<int, 1>(n);
  int info = 0;
  api::ComplexType<float> worksize;
  magma_cheevx_gpu(jobzEnum, rangeEnum, uploEnum, n, reinterpret_cast<magmaFloatComplex*>(a), lda,
                   0, 0, 0, 0, abstol, m, wHost.data(),
                   reinterpret_cast<magmaFloatComplex*>(z.data()), ldz,
                   reinterpret_cast<magmaFloatComplex*>(wA.data()), n,
                   reinterpret_cast<magmaFloatComplex*>(wZ.data()), n,
                   reinterpret_cast<magmaFloatComplex*>(&worksize), -1, rwork.data(), iwork.data(),
                   ifail.data(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n) lwork = 2 * n;
  auto work = queue.create_host_array<api::ComplexType<float>, 1>(lwork);
  magma_cheevx_gpu(jobzEnum, rangeEnum, uploEnum, n, reinterpret_cast<magmaFloatComplex*>(a), lda,
                   0, 0, 0, 0, abstol, m, wHost.data(),
                   reinterpret_cast<magmaFloatComplex*>(z.data()), ldz,
                   reinterpret_cast<magmaFloatComplex*>(wA.data()), n,
                   reinterpret_cast<magmaFloatComplex*>(wZ.data()), n,
                   reinterpret_cast<magmaFloatComplex*>(work.data()), lwork, rwork.data(),
                   iwork.data(), ifail.data(), &info);

  if (info != 0) throw EigensolverError();

  api::memcpy_async(w, wHost.data(), (*m) * sizeof(float), api::flag::MemcpyHostToDevice,
                    queue.stream());
  api::memcpy_2d_async(a, lda * sizeof(api::ComplexType<float>), z.data(),
                       ldz * sizeof(api::ComplexType<float>), n * sizeof(api::ComplexType<float>),
                       *m, api::flag::MemcpyDeviceToDevice, queue.stream());
  queue.sync();

#else

  int lwork = 0;
  if (cusolverDnCheevd_bufferSize(queue.solver_handle(), jobzEnum, uploEnum, n, a, lda, w,
                                  &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace = queue.create_device_array<api::ComplexType<float>, 1>(std::size_t(lwork));
  auto devInfo = queue.create_device_array<int, 1>(1);
  api::memset_async(devInfo.data(), 0, sizeof(int), queue.stream());
  if (cusolverDnCheevd(queue.solver_handle(), jobzEnum, uploEnum, n, a, lda, w, workspace.data(),
                       lwork, devInfo.data()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  int hostInfo;
  api::memcpy_async(&hostInfo, devInfo.data(), sizeof(int), api::flag::MemcpyDeviceToHost,
                    queue.stream());
  queue.sync();
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}

auto solve(Queue& queue, char jobz, char uplo, int n, api::ComplexType<double>* a, int lda,
           double* w) -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto uploEnum = convert_uplo(uplo);

#ifdef BIPP_MAGMA
  queue.sync();
  auto rangeEnum = convert_range('A');
  const double abstol = 2 * host::lapack::dlamch('S');

  auto z = queue.create_device_array<api::ComplexType<double>, 1>(n * n);
  int ldz = n;

  auto wHost = queue.create_pinned_array<double, 1>(n);
  auto wA = queue.create_host_array<api::ComplexType<double>, 1>(n * n);
  auto wZ = queue.create_host_array<api::ComplexType<double>, 1>(n * n);
  auto rwork = queue.create_host_array<double, 1>(7 * n);
  auto iwork = queue.create_host_array<int, 1>(5 * n);
  auto ifail = queue.create_host_array<int, 1>(n);
  int info = 0;
  api::ComplexType<double> worksize;
  magma_zheevx_gpu(jobzEnum, rangeEnum, uploEnum, n, reinterpret_cast<magmaDoubleComplex*>(a), lda,
                   0, 0, 0, 0, abstol, m, wHost.data(),
                   reinterpret_cast<magmaDoubleComplex*>(z.data()), ldz,
                   reinterpret_cast<magmaDoubleComplex*>(wA.data()), n,
                   reinterpret_cast<magmaDoubleComplex*>(wZ.data()), n,
                   reinterpret_cast<magmaDoubleComplex*>(&worksize), -1, rwork.data(), iwork.data(),
                   ifail.data(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n) lwork = 2 * n;
  auto work = queue.create_host_array<api::ComplexType<double>, 1>(lwork);
  magma_zheevx_gpu(jobzEnum, rangeEnum, uploEnum, n, reinterpret_cast<magmaDoubleComplex*>(a), lda,
                   0, 0, 0, 0, abstol, m, wHost.data(),
                   reinterpret_cast<magmaDoubleComplex*>(z.data()), ldz,
                   reinterpret_cast<magmaDoubleComplex*>(wA.data()), n,
                   reinterpret_cast<magmaDoubleComplex*>(wZ.data()), n,
                   reinterpret_cast<magmaDoubleComplex*>(work.data()), lwork, rwork.data(),
                   iwork.data(), ifail.data(), &info);

  if (info != 0) throw EigensolverError();

  api::memcpy_async(w, wHost.data(), (*m) * sizeof(double), api::flag::MemcpyHostToDevice,
                    queue.stream());
  api::memcpy_2d_async(a, lda * sizeof(api::ComplexType<double>), z.data(),
                       ldz * sizeof(api::ComplexType<double>), n * sizeof(api::ComplexType<double>),
                       *m, api::flag::MemcpyDeviceToDevice, queue.stream());
  queue.sync();

#else

  int lwork = 0;
  if (cusolverDnZheevd_bufferSize(queue.solver_handle(), jobzEnum, uploEnum, n, a, lda, w,
                                  &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace = queue.create_device_array<api::ComplexType<double>, 1>(std::size_t(lwork));
  auto devInfo = queue.create_device_array<int, 1>(1);
  api::memset_async(devInfo.data(), 0, sizeof(int), queue.stream());
  if (cusolverDnZheevd(queue.solver_handle(), jobzEnum, uploEnum, n, a, lda, w, workspace.data(),
                       lwork, devInfo.data()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  int hostInfo;
  api::memcpy_async(&hostInfo, devInfo.data(), sizeof(int), api::flag::MemcpyDeviceToHost,
                    queue.stream());
  queue.sync();
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}

auto solve(Queue& queue, char jobz, char uplo, int n, api::ComplexType<float>* a, int lda,
           api::ComplexType<float>* b, int ldb, float* w) -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto uploEnum = convert_uplo(uplo);

#ifdef BIPP_MAGMA
  queue.sync();
  auto rangeEnum = convert_range('A');
  auto aHost = queue.create_pinned_array<api::ComplexType<float>, 1>(n * n);
  auto bHost = queue.create_pinned_array<api::ComplexType<float>, 1>(n * n);
  auto zHost = queue.create_pinned_array<api::ComplexType<float>, 1>(n * n);
  api::memcpy_2d_async(aHost.data(), n * sizeof(api::ComplexType<float>), a,
                       lda * sizeof(api::ComplexType<float>), n * sizeof(api::ComplexType<float>),
                       n, api::flag::MemcpyDeviceToHost, queue.stream());
  api::memcpy_2d_async(bHost.data(), n * sizeof(api::ComplexType<float>), b,
                       ldb * sizeof(api::ComplexType<float>), n * sizeof(api::ComplexType<float>),
                       n, api::flag::MemcpyDeviceToHost, queue.stream());
  api::stream_synchronize(queue.stream());

  const float abstol = 2 * host::lapack::slamch('S');

  int ldz = n;

  auto wHost = queue.create_pinned_array<float, 1>(n);
  auto rwork = queue.create_host_array<float, 1>(7 * n);
  auto iwork = queue.create_host_array<int, 1>(5 * n);
  auto ifail = queue.create_host_array<int, 1>(n);
  int info = 0;
  api::ComplexType<float> worksize;
  magma_chegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaFloatComplex*>(aHost.data()), n,
               reinterpret_cast<magmaFloatComplex*>(bHost.data()), n, 0, 0, 0, 0, abstol, m,
               wHost.data(), reinterpret_cast<magmaFloatComplex*>(zHost.data()), ldz,
               reinterpret_cast<magmaFloatComplex*>(&worksize), -1, rwork.data(), iwork.data(),
               ifail.data(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n) lwork = 2 * n;
  auto work = queue.create_host_array<api::ComplexType<float>, 1>(lwork);
  magma_chegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaFloatComplex*>(aHost.data()), n,
               reinterpret_cast<magmaFloatComplex*>(bHost.data()), n, 0, 0, 0, 0, abstol, m,
               wHost.data(), reinterpret_cast<magmaFloatComplex*>(zHost.data()), ldz,
               reinterpret_cast<magmaFloatComplex*>(work.data()), lwork, rwork.data(), iwork.data(),
               ifail.data(), &info);

  if (info != 0) throw EigensolverError();

  api::memcpy_async(w, wHost.data(), (*m) * sizeof(float), api::flag::MemcpyHostToDevice,
                    queue.stream());
  api::memcpy_2d_async(a, lda * sizeof(api::ComplexType<float>), zHost.data(),
                       ldz * sizeof(api::ComplexType<float>), n * sizeof(api::ComplexType<float>),
                       *m, api::flag::MemcpyDeviceToHost, queue.stream());
  queue.sync();

#else

  int lwork = 0;
  if (cusolverDnChegvd_bufferSize(queue.solver_handle(), CUSOLVER_EIG_TYPE_1, jobzEnum, uploEnum, n,
                                  a, lda, b, ldb, w, &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace = queue.create_device_array<api::ComplexType<float>, 1>(std::size_t(lwork));
  auto devInfo = queue.create_device_array<int, 1>(1);
  api::memset_async(devInfo.data(), 0, sizeof(int), queue.stream());
  if (cusolverDnChegvd(queue.solver_handle(), CUSOLVER_EIG_TYPE_1, jobzEnum, uploEnum, n, a, lda, b,
                       ldb, w, workspace.data(), lwork, devInfo.data()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  int hostInfo;
  api::memcpy_async(&hostInfo, devInfo.data(), sizeof(int), api::flag::MemcpyDeviceToHost,
                    queue.stream());
  queue.sync();
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}

auto solve(Queue& queue, char jobz, char uplo, int n, api::ComplexType<double>* a, int lda,
           api::ComplexType<double>* b, int ldb, double* w) -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto uploEnum = convert_uplo(uplo);

#ifdef BIPP_MAGMA
  queue.sync();
  auto rangeEnum = convert_range('A');
  auto aHost = queue.create_pinned_array<api::ComplexType<double>, 1>(n * n);
  auto bHost = queue.create_pinned_array<api::ComplexType<double>, 1>(n * n);
  auto zHost = queue.create_pinned_array<api::ComplexType<double>, 1>(n * n);
  api::memcpy_2d_async(aHost.data(), n * sizeof(api::ComplexType<double>), a,
                       lda * sizeof(api::ComplexType<double>), n * sizeof(api::ComplexType<double>),
                       n, api::flag::MemcpyDeviceToHost, queue.stream());
  api::memcpy_2d_async(bHost.data(), n * sizeof(api::ComplexType<double>), b,
                       ldb * sizeof(api::ComplexType<double>), n * sizeof(api::ComplexType<double>),
                       n, api::flag::MemcpyDeviceToHost, queue.stream());
  api::stream_synchronize(queue.stream());

  const double abstol = 2 * host::lapack::dlamch('S');

  int ldz = n;

  auto wHost = queue.create_pinned_array<double, 1>(n);
  auto rwork = queue.create_host_array<double, 1>(7 * n);
  auto iwork = queue.create_host_array<int, 1>(5 * n);
  auto ifail = queue.create_host_array<int, 1>(n);
  int info = 0;
  api::ComplexType<double> worksize;
  magma_zhegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaDoubleComplex*>(aHost.data()), n,
               reinterpret_cast<magmaDoubleComplex*>(bHost.data()), n, 0, 0, 0, 0, abstol, m,
               wHost.data(), reinterpret_cast<magmaDoubleComplex*>(zHost.data()), ldz,
               reinterpret_cast<magmaDoubleComplex*>(&worksize), -1, rwork.data(), iwork.data(),
               ifail.data(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n) lwork = 2 * n;
  auto work = queue.create_host_array<api::ComplexType<double>, 1>(lwork);
  magma_zhegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaDoubleComplex*>(aHost.data()), n,
               reinterpret_cast<magmaDoubleComplex*>(bHost.data()), n, 0, 0, 0, 0, abstol, m,
               wHost.data(), reinterpret_cast<magmaDoubleComplex*>(zHost.data()), ldz,
               reinterpret_cast<magmaDoubleComplex*>(work.data()), lwork, rwork.data(),
               iwork.data(), ifail.data(), &info);

  if (info != 0) throw EigensolverError();

  api::memcpy_async(w, wHost.data(), (*m) * sizeof(double), api::flag::MemcpyHostToDevice,
                    queue.stream());
  api::memcpy_2d_async(a, lda * sizeof(api::ComplexType<double>), zHost.data(),
                       ldz * sizeof(api::ComplexType<double>), n * sizeof(api::ComplexType<double>),
                       *m, api::flag::MemcpyDeviceToHost, queue.stream());
  queue.sync();

#else

  int lwork = 0;
  if (cusolverDnZhegvd_bufferSize(queue.solver_handle(), CUSOLVER_EIG_TYPE_1, jobzEnum, uploEnum, n,
                                  a, lda, b, ldb, w, &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace = queue.create_device_array<api::ComplexType<double>, 1>(std::size_t(lwork));
  auto devInfo = queue.create_device_array<int, 1>(1);
  api::memset_async(devInfo.data(), 0, sizeof(int), queue.stream());
  if (cusolverDnZhegvd(queue.solver_handle(), CUSOLVER_EIG_TYPE_1, jobzEnum, uploEnum, n, a, lda, b,
                       ldb, w, workspace.data(), lwork, devInfo.data()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  int hostInfo;
  api::memcpy_async(&hostInfo, devInfo.data(), sizeof(int), api::flag::MemcpyDeviceToHost,
                    queue.stream());
  queue.sync();
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}
}  // namespace eigensolver

}  // namespace gpu
}  // namespace bipp
