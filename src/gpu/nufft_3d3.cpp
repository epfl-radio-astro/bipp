#include "gpu/nufft_3d3.hpp"

#include <cufinufft.h>

#include <complex>
#include <functional>
#include <memory>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <>
Nufft3d3<double>::Nufft3d3(int iflag, double tol, std::size_t numTrans, std::size_t M,
                           const double* x, const double* y, const double* z, std::size_t N,
                           const double* s, const double* t, const double* u) {
  cufinufft_opts opts;
  cufinufft_default_opts(3, 3, &opts);
  cufinufft_plan p;
  if (cufinufft_makeplan(3, 3, nullptr, iflag, numTrans, tol, numTrans, &p, &opts))
    throw FiNUFFTError();

  plan_ = planType(new cufinufft_plan(p), [](void* ptr) {
    auto castPtr = reinterpret_cast<cufinufft_plan*>(ptr);
    cufinufft_destroy(*castPtr);
    delete castPtr;
  });

  if (cufinufft_setpts(M, const_cast<double*>(x), const_cast<double*>(y), const_cast<double*>(z), N,
                       const_cast<double*>(s), const_cast<double*>(t), const_cast<double*>(u), p))
    throw FiNUFFTError();
}

template <>
void Nufft3d3<double>::execute(const api::ComplexType<double>* cj, api::ComplexType<double>* fk) {
  cufinufft_execute(const_cast<api::ComplexType<double>*>(cj), fk,
                    *reinterpret_cast<const cufinufft_plan*>(plan_.get()));
}

template <>
Nufft3d3<float>::Nufft3d3(int iflag, float tol, std::size_t numTrans, std::size_t M, const float* x,
                          const float* y, const float* z, std::size_t N, const float* s,
                          const float* t, const float* u) {
  cufinufft_opts opts;
  cufinufftf_default_opts(3, 3, &opts);
  cufinufftf_plan p;
  if (cufinufftf_makeplan(3, 3, nullptr, iflag, numTrans, tol, numTrans, &p, &opts))
    throw FiNUFFTError();

  plan_ = planType(new cufinufftf_plan(p), [](void* ptr) {
    auto castPtr = reinterpret_cast<cufinufftf_plan*>(ptr);
    cufinufftf_destroy(*castPtr);
    delete castPtr;
  });

  if (cufinufftf_setpts(M, const_cast<float*>(x), const_cast<float*>(y), const_cast<float*>(z), N,
                        const_cast<float*>(s), const_cast<float*>(t), const_cast<float*>(u), p))
    throw FiNUFFTError();
}

template <>
void Nufft3d3<float>::execute(const api::ComplexType<float>* cj, api::ComplexType<float>* fk) {
  cufinufftf_execute(const_cast<api::ComplexType<float>*>(cj), fk,
                     *reinterpret_cast<const cufinufftf_plan*>(plan_.get()));
}

}  // namespace gpu
}  // namespace bipp
