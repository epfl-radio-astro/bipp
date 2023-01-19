#include "gpu/util/solver_api.hpp"

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"
#include "host/lapack_api.hpp"
#include "memory/buffer.hpp"

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

auto solve(ContextInternal& ctx, char jobz, char range, char uplo, int n,
           api::ComplexType<float>* a, int lda, float vl, float vu, int il, int iu, int* m,
           float* w) -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto rangeEnum = convert_range(range);
  auto uploEnum = convert_uplo(uplo);

#ifdef BIPP_MAGMA
  api::stream_synchronize(ctx.gpu_queue().stream());
  const float abstol = 2 * lapack::slamch('S');

  auto z = ctx.gpu_queue().create_device_buffer<api::ComplexType<float>>(n * n);
  int ldz = n;

  auto wHost = ctx.gpu_queue().create_pinned_buffer<float>(n);
  auto wA = ctx.gpu_queue().create_host_buffer<api::ComplexType<float>>(n * n);
  auto wZ = ctx.gpu_queue().create_host_buffer<api::ComplexType<float>>(n * n);
  auto rwork = ctx.gpu_queue().create_host_buffer<float>(7 * n);
  auto iwork = ctx.gpu_queue().create_host_buffer<int>(5 * n);
  auto ifail = ctx.gpu_queue().create_host_buffer<int>(n);
  int info = 0;
  api::ComplexType<float> worksize;
  magma_cheevx_gpu(jobzEnum, rangeEnum, uploEnum, n, reinterpret_cast<magmaFloatComplex*>(a), lda,
                   vl, vu, il, iu, abstol, m, wHost.get(),
                   reinterpret_cast<magmaFloatComplex*>(z.get()), ldz,
                   reinterpret_cast<magmaFloatComplex*>(wA.get()), n,
                   reinterpret_cast<magmaFloatComplex*>(wZ.get()), n,
                   reinterpret_cast<magmaFloatComplex*>(&worksize), -1, rwork.get(), iwork.get(),
                   ifail.get(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n) lwork = 2 * n;
  auto work = ctx.gpu_queue().create_host_buffer<api::ComplexType<float>>(lwork);
  magma_cheevx_gpu(jobzEnum, rangeEnum, uploEnum, n, reinterpret_cast<magmaFloatComplex*>(a), lda,
                   vl, vu, il, iu, abstol, m, wHost.get(),
                   reinterpret_cast<magmaFloatComplex*>(z.get()), ldz,
                   reinterpret_cast<magmaFloatComplex*>(wA.get()), n,
                   reinterpret_cast<magmaFloatComplex*>(wZ.get()), n,
                   reinterpret_cast<magmaFloatComplex*>(work.get()), lwork, rwork.get(),
                   iwork.get(), ifail.get(), &info);

  if (info != 0) throw EigensolverError();

  api::memcpy_async(w, wHost.get(), (*m) * sizeof(float), api::flag::MemcpyHostToDevice,
                    ctx.gpu_queue().stream());
  api::memcpy_2d_async(a, lda * sizeof(api::ComplexType<float>), z.get(),
                       ldz * sizeof(api::ComplexType<float>), n * sizeof(api::ComplexType<float>),
                       *m, api::flag::MemcpyDeviceToDevice, ctx.gpu_queue().stream());
  api::stream_synchronize(ctx.gpu_queue().stream());

#else

  int lwork = 0;
  if (cusolverDnCheevdx_bufferSize(ctx.gpu_queue().solver_handle(), jobzEnum, rangeEnum, uploEnum,
                                   n, a, lda, vl, vu, il, iu, nullptr, nullptr,
                                   &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace = ctx.gpu_queue().create_device_buffer<api::ComplexType<float>>(lwork);
  auto devInfo = ctx.gpu_queue().create_device_buffer<int>(1);
  if (cusolverDnCheevdx(ctx.gpu_queue().solver_handle(), jobzEnum, rangeEnum, uploEnum, n, a, lda,
                        vl, vu, il, iu, m, w, workspace.get(), lwork,
                        devInfo.get()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  int hostInfo;
  api::memcpy_async(&hostInfo, devInfo.get(), sizeof(int), api::flag::MemcpyDeviceToHost,
                    ctx.gpu_queue().stream());
  api::stream_synchronize(ctx.gpu_queue().stream());
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}

auto solve(ContextInternal& ctx, char jobz, char range, char uplo, int n,
           api::ComplexType<double>* a, int lda, double vl, double vu, int il, int iu, int* m,
           double* w) -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto rangeEnum = convert_range(range);
  auto uploEnum = convert_uplo(uplo);

#ifdef BIPP_MAGMA
  api::stream_synchronize(ctx.gpu_queue().stream());
  const double abstol = 2 * lapack::dlamch('S');

  auto z = ctx.gpu_queue().create_device_buffer<api::ComplexType<double>>(n * n);
  int ldz = n;

  auto wHost = ctx.gpu_queue().create_pinned_buffer<double>(n);
  auto wA = ctx.gpu_queue().create_host_buffer<api::ComplexType<double>>(n * n);
  auto wZ = ctx.gpu_queue().create_host_buffer<api::ComplexType<double>>(n * n);
  auto rwork = ctx.gpu_queue().create_host_buffer<double>(7 * n);
  auto iwork = ctx.gpu_queue().create_host_buffer<int>(5 * n);
  auto ifail = ctx.gpu_queue().create_host_buffer<int>(n);
  int info = 0;
  api::ComplexType<double> worksize;
  magma_zheevx_gpu(jobzEnum, rangeEnum, uploEnum, n, reinterpret_cast<magmaDoubleComplex*>(a), lda,
                   vl, vu, il, iu, abstol, m, wHost.get(),
                   reinterpret_cast<magmaDoubleComplex*>(z.get()), ldz,
                   reinterpret_cast<magmaDoubleComplex*>(wA.get()), n,
                   reinterpret_cast<magmaDoubleComplex*>(wZ.get()), n,
                   reinterpret_cast<magmaDoubleComplex*>(&worksize), -1, rwork.get(), iwork.get(),
                   ifail.get(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n) lwork = 2 * n;
  auto work = ctx.gpu_queue().create_host_buffer<api::ComplexType<double>>(lwork);
  magma_zheevx_gpu(jobzEnum, rangeEnum, uploEnum, n, reinterpret_cast<magmaDoubleComplex*>(a), lda,
                   vl, vu, il, iu, abstol, m, wHost.get(),
                   reinterpret_cast<magmaDoubleComplex*>(z.get()), ldz,
                   reinterpret_cast<magmaDoubleComplex*>(wA.get()), n,
                   reinterpret_cast<magmaDoubleComplex*>(wZ.get()), n,
                   reinterpret_cast<magmaDoubleComplex*>(work.get()), lwork, rwork.get(),
                   iwork.get(), ifail.get(), &info);

  if (info != 0) throw EigensolverError();

  api::memcpy_async(w, wHost.get(), (*m) * sizeof(double), api::flag::MemcpyHostToDevice,
                    ctx.gpu_queue().stream());
  api::memcpy_2d_async(a, lda * sizeof(api::ComplexType<double>), z.get(),
                       ldz * sizeof(api::ComplexType<double>), n * sizeof(api::ComplexType<double>),
                       *m, api::flag::MemcpyDeviceToDevice, ctx.gpu_queue().stream());
  api::stream_synchronize(ctx.gpu_queue().stream());

#else

  int lwork = 0;
  if (cusolverDnZheevdx_bufferSize(ctx.gpu_queue().solver_handle(), jobzEnum, rangeEnum, uploEnum,
                                   n, a, lda, vl, vu, il, iu, nullptr, nullptr,
                                   &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace = ctx.gpu_queue().create_device_buffer<api::ComplexType<double>>(lwork);
  auto devInfo = ctx.gpu_queue().create_device_buffer<int>(1);
  // make sure info is always 0. Second entry might not be set otherwise.
  api::memset_async(devInfo.get(), 0, sizeof(int), ctx.gpu_queue().stream());
  if (cusolverDnZheevdx(ctx.gpu_queue().solver_handle(), jobzEnum, rangeEnum, uploEnum, n, a, lda,
                        vl, vu, il, iu, m, w, workspace.get(), lwork,
                        devInfo.get()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();
  int hostInfo;
  api::memcpy_async(&hostInfo, devInfo.get(), sizeof(int), api::flag::MemcpyDeviceToHost,
                    ctx.gpu_queue().stream());
  api::stream_synchronize(ctx.gpu_queue().stream());
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}

auto solve(ContextInternal& ctx, char jobz, char range, char uplo, int n,
           api::ComplexType<float>* a, int lda, api::ComplexType<float>* b, int ldb, float vl,
           float vu, int il, int iu, int* m, float* w) -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto rangeEnum = convert_range(range);
  auto uploEnum = convert_uplo(uplo);

#ifdef BIPP_MAGMA
  auto aHost = ctx.gpu_queue().create_pinned_buffer<api::ComplexType<float>>(n * n);
  auto bHost = ctx.gpu_queue().create_pinned_buffer<api::ComplexType<float>>(n * n);
  auto zHost = ctx.gpu_queue().create_pinned_buffer<api::ComplexType<float>>(n * n);
  api::memcpy_2d_async(aHost.get(), n * sizeof(api::ComplexType<float>), a,
                       lda * sizeof(api::ComplexType<float>), n * sizeof(api::ComplexType<float>),
                       n, api::flag::MemcpyDeviceToHost, ctx.gpu_queue().stream());
  api::memcpy_2d_async(bHost.get(), n * sizeof(api::ComplexType<float>), b,
                       ldb * sizeof(api::ComplexType<float>), n * sizeof(api::ComplexType<float>),
                       n, api::flag::MemcpyDeviceToHost, ctx.gpu_queue().stream());
  api::stream_synchronize(ctx.gpu_queue().stream());

  const float abstol = 2 * lapack::slamch('S');

  int ldz = n;

  auto wHost = ctx.gpu_queue().create_pinned_buffer<float>(n);
  auto rwork = ctx.gpu_queue().create_host_buffer<float>(7 * n);
  auto iwork = ctx.gpu_queue().create_host_buffer<int>(5 * n);
  auto ifail = ctx.gpu_queue().create_host_buffer<int>(n);
  int info = 0;
  api::ComplexType<float> worksize;
  magma_chegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaFloatComplex*>(aHost.get()), n,
               reinterpret_cast<magmaFloatComplex*>(bHost.get()), n, vl, vu, il, iu, abstol, m,
               wHost.get(), reinterpret_cast<magmaFloatComplex*>(zHost.get()), ldz,
               reinterpret_cast<magmaFloatComplex*>(&worksize), -1, rwork.get(), iwork.get(),
               ifail.get(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n) lwork = 2 * n;
  auto work = ctx.gpu_queue().create_host_buffer<api::ComplexType<float>>(lwork);
  magma_chegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaFloatComplex*>(aHost.get()), n,
               reinterpret_cast<magmaFloatComplex*>(bHost.get()), n, vl, vu, il, iu, abstol, m,
               wHost.get(), reinterpret_cast<magmaFloatComplex*>(zHost.get()), ldz,
               reinterpret_cast<magmaFloatComplex*>(work.get()), lwork, rwork.get(), iwork.get(),
               ifail.get(), &info);

  if (info != 0) throw EigensolverError();

  api::memcpy_async(w, wHost.get(), (*m) * sizeof(float), api::flag::MemcpyHostToDevice,
                    ctx.gpu_queue().stream());
  api::memcpy_2d_async(a, lda * sizeof(api::ComplexType<float>), zHost.get(),
                       ldz * sizeof(api::ComplexType<float>), n * sizeof(api::ComplexType<float>),
                       *m, api::flag::MemcpyDeviceToHost, ctx.gpu_queue().stream());
  api::stream_synchronize(ctx.gpu_queue().stream());

#else

  int lwork = 0;
  if (cusolverDnChegvdx_bufferSize(ctx.gpu_queue().solver_handle(), CUSOLVER_EIG_TYPE_1, jobzEnum,
                                   rangeEnum, uploEnum, n, a, lda, b, ldb, vl, vu, il, iu, nullptr,
                                   nullptr, &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace = ctx.gpu_queue().create_device_buffer<api::ComplexType<float>>(lwork);
  auto devInfo = ctx.gpu_queue().create_device_buffer<int>(2);
  if (cusolverDnChegvdx(ctx.gpu_queue().solver_handle(), CUSOLVER_EIG_TYPE_1, jobzEnum, rangeEnum,
                        uploEnum, n, a, lda, b, ldb, vl, vu, il, iu, m, w, workspace.get(), lwork,
                        devInfo.get()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();
  int hostInfo;
  api::memcpy_async(&hostInfo, devInfo.get(), sizeof(int), api::flag::MemcpyDeviceToHost,
                    ctx.gpu_queue().stream());
  api::stream_synchronize(ctx.gpu_queue().stream());
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}

auto solve(ContextInternal& ctx, char jobz, char range, char uplo, int n,
           api::ComplexType<double>* a, int lda, api::ComplexType<double>* b, int ldb, double vl,
           double vu, int il, int iu, int* m, double* w) -> void {
  auto jobzEnum = convert_jobz(jobz);
  auto rangeEnum = convert_range(range);
  auto uploEnum = convert_uplo(uplo);

#ifdef BIPP_MAGMA
  auto aHost = ctx.gpu_queue().create_pinned_buffer<api::ComplexType<double>>(n * n);
  auto bHost = ctx.gpu_queue().create_pinned_buffer<api::ComplexType<double>>(n * n);
  auto zHost = ctx.gpu_queue().create_pinned_buffer<api::ComplexType<double>>(n * n);
  api::memcpy_2d_async(aHost.get(), n * sizeof(api::ComplexType<double>), a,
                       lda * sizeof(api::ComplexType<double>), n * sizeof(api::ComplexType<double>),
                       n, api::flag::MemcpyDeviceToHost, ctx.gpu_queue().stream());
  api::memcpy_2d_async(bHost.get(), n * sizeof(api::ComplexType<double>), b,
                       ldb * sizeof(api::ComplexType<double>), n * sizeof(api::ComplexType<double>),
                       n, api::flag::MemcpyDeviceToHost, ctx.gpu_queue().stream());
  api::stream_synchronize(ctx.gpu_queue().stream());

  const double abstol = 2 * lapack::dlamch('S');

  int ldz = n;

  auto wHost = ctx.gpu_queue().create_pinned_buffer<double>(n);
  auto rwork = ctx.gpu_queue().create_host_buffer<double>(7 * n);
  auto iwork = ctx.gpu_queue().create_host_buffer<int>(5 * n);
  auto ifail = ctx.gpu_queue().create_host_buffer<int>(n);
  int info = 0;
  api::ComplexType<double> worksize;
  magma_zhegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaDoubleComplex*>(aHost.get()), n,
               reinterpret_cast<magmaDoubleComplex*>(bHost.get()), n, vl, vu, il, iu, abstol, m,
               wHost.get(), reinterpret_cast<magmaDoubleComplex*>(zHost.get()), ldz,
               reinterpret_cast<magmaDoubleComplex*>(&worksize), -1, rwork.get(), iwork.get(),
               ifail.get(), &info);
  int lwork = static_cast<int>(worksize.x);
  if (lwork < 2 * n) lwork = 2 * n;
  auto work = ctx.gpu_queue().create_host_buffer<api::ComplexType<double>>(lwork);
  magma_zhegvx(1, jobzEnum, rangeEnum, uploEnum, n,
               reinterpret_cast<magmaDoubleComplex*>(aHost.get()), n,
               reinterpret_cast<magmaDoubleComplex*>(bHost.get()), n, vl, vu, il, iu, abstol, m,
               wHost.get(), reinterpret_cast<magmaDoubleComplex*>(zHost.get()), ldz,
               reinterpret_cast<magmaDoubleComplex*>(work.get()), lwork, rwork.get(), iwork.get(),
               ifail.get(), &info);

  if (info != 0) throw EigensolverError();

  api::memcpy_async(w, wHost.get(), (*m) * sizeof(double), api::flag::MemcpyHostToDevice,
                    ctx.gpu_queue().stream());
  api::memcpy_2d_async(a, lda * sizeof(api::ComplexType<double>), zHost.get(),
                       ldz * sizeof(api::ComplexType<double>), n * sizeof(api::ComplexType<double>),
                       *m, api::flag::MemcpyDeviceToHost, ctx.gpu_queue().stream());
  api::stream_synchronize(ctx.gpu_queue().stream());
#else
  int lwork = 0;
  if (cusolverDnZhegvdx_bufferSize(ctx.gpu_queue().solver_handle(), CUSOLVER_EIG_TYPE_1, jobzEnum,
                                   rangeEnum, uploEnum, n, a, lda, b, ldb, vl, vu, il, iu, nullptr,
                                   nullptr, &lwork) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();

  auto workspace = ctx.gpu_queue().create_device_buffer<api::ComplexType<double>>(lwork);
  auto devInfo = ctx.gpu_queue().create_device_buffer<int>(1);
  if (cusolverDnZhegvdx(ctx.gpu_queue().solver_handle(), CUSOLVER_EIG_TYPE_1, jobzEnum, rangeEnum,
                        uploEnum, n, a, lda, b, ldb, vl, vu, il, iu, m, w, workspace.get(), lwork,
                        devInfo.get()) != CUSOLVER_STATUS_SUCCESS)
    throw EigensolverError();
  int hostInfo;
  api::memcpy_async(&hostInfo, devInfo.get(), sizeof(int), api::flag::MemcpyDeviceToHost,
                    ctx.gpu_queue().stream());
  api::stream_synchronize(ctx.gpu_queue().stream());
  if (hostInfo) {
    throw EigensolverError();
  }
#endif
}
}  // namespace eigensolver

}  // namespace gpu
}  // namespace bipp
