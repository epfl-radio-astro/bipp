#include <complex>
#include <cstddef>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/eigensolver.hpp"
#include "gpu/kernels/gram.hpp"
#include "gpu/util/blas_api.hpp"
#include "gpu/util/runtime_api.hpp"
#include "memory/buffer.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto gram_matrix(ContextInternal& ctx, std::size_t m, std::size_t n, const api::ComplexType<T>* w,
                 std::size_t ldw, const T* xyz, std::size_t ldxyz, T wl, api::ComplexType<T>* g,
                 std::size_t ldg) -> void {
  using ComplexType = api::ComplexType<T>;

  auto& queue = ctx.gpu_queue();
  // Syncronize with default stream.
  queue.sync_with_stream(nullptr);
  // syncronize with stream to be synchronous with host before exiting
  auto syncGuard = queue.sync_guard();

  auto baseD = queue.create_device_buffer<ComplexType>(m * m);

  {
    auto xyzD = queue.create_device_buffer<T>(3 * m);
    api::memcpy_2d_async(xyzD.get(), m * sizeof(T), xyz, ldxyz * sizeof(T), m * sizeof(T), 3,
                         api::flag::MemcpyDefault, queue.stream());
    gram(queue, m, xyzD.get(), xyzD.get() + m, xyzD.get() + 2 * m, wl, baseD.get(), m);
  }

  auto wD = queue.create_device_buffer<ComplexType>(m * n);
  auto cD = queue.create_device_buffer<ComplexType>(m * n);
  api::memcpy_2d_async(wD.get(), m * sizeof(ComplexType), w, ldw * sizeof(ComplexType),
                       m * sizeof(ComplexType), n, api::flag::MemcpyDefault, queue.stream());
  ComplexType alpha{1, 0};
  ComplexType beta{0, 0};
  api::blas::symm(queue.blas_handle(), api::blas::side::left, api::blas::fill::lower, m, n, &alpha,
                  baseD.get(), m, wD.get(), m, &beta, cD.get(), m);
  auto gD = queue.create_device_buffer<ComplexType>(n * n);
  api::blas::gemm(queue.blas_handle(), api::blas::operation::ConjugateTranspose,
                  api::blas::operation::None, n, n, m, &alpha, wD.get(), m, cD.get(), m, &beta,
                  gD.get(), n);

  api::memcpy_2d_async(g, ldg * sizeof(ComplexType), gD.get(), n * sizeof(ComplexType),
                       n * sizeof(ComplexType), n, api::flag::MemcpyDefault, queue.stream());
}

template auto gram_matrix<float>(ContextInternal& ctx, std::size_t m, std::size_t n,
                                 const api::ComplexType<float>* w, std::size_t ldw,
                                 const float* xyz, std::size_t ldxyz, float wl,
                                 api::ComplexType<float>* g, std::size_t ldg) -> void;

template auto gram_matrix<double>(ContextInternal& ctx, std::size_t m, std::size_t n,
                                  const api::ComplexType<double>* w, std::size_t ldw,
                                  const double* xyz, std::size_t ldxyz, double wl,
                                  api::ComplexType<double>* g, std::size_t ldg) -> void;

}  // namespace gpu
}  // namespace bipp
