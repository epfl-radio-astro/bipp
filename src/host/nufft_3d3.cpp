#include "host/nufft_3d3.hpp"

#include <finufft.h>

#include <complex>
#include <functional>
#include <memory>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"

namespace bipp {
namespace host {

template <>
Nufft3d3<double>::Nufft3d3(int iflag, double tol, std::size_t numTrans, std::size_t M,
                           const double* x, const double* y, const double* z, std::size_t N,
                           const double* s, const double* t, const double* u) {
  finufft_opts opts;
  finufft_default_opts(&opts);
  opts.upsampfac = 2.0; // must be set to prevent automatic selection based on tol
  finufft_plan p;
  if (finufft_makeplan(3, 3, nullptr, iflag, numTrans, tol, &p, &opts)) throw FiNUFFTError();

  plan_ = planType(new finufft_plan(p), [](void* ptr) {
    auto castPtr = reinterpret_cast<finufft_plan*>(ptr);
    finufft_destroy(*castPtr);
    delete castPtr;
  });

  finufft_setpts(p, M, const_cast<double*>(x), const_cast<double*>(y), const_cast<double*>(z), N,
                 const_cast<double*>(s), const_cast<double*>(t), const_cast<double*>(u));
}

template <>
void Nufft3d3<double>::execute(const std::complex<double>* cj, std::complex<double>* fk) {
  finufft_execute(*reinterpret_cast<const finufft_plan*>(plan_.get()),
                  const_cast<std::complex<double>*>(cj), fk);
}

template <>
Nufft3d3<float>::Nufft3d3(int iflag, float tol, std::size_t numTrans, std::size_t M, const float* x,
                          const float* y, const float* z, std::size_t N, const float* s,
                          const float* t, const float* u) {
  finufft_opts opts;
  finufftf_default_opts(&opts);
  opts.upsampfac = 2.0; // must be set to prevent automatic selection based on tol
  finufftf_plan p;
  if (finufftf_makeplan(3, 3, nullptr, iflag, numTrans, tol, &p, &opts)) throw FiNUFFTError();

  plan_ = planType(new finufftf_plan(p), [](void* ptr) {
    auto castPtr = reinterpret_cast<finufftf_plan*>(ptr);
    finufftf_destroy(*castPtr);
    delete castPtr;
  });

  if (finufftf_setpts(p, M, const_cast<float*>(x), const_cast<float*>(y), const_cast<float*>(z), N,
                      const_cast<float*>(s), const_cast<float*>(t), const_cast<float*>(u)))
    throw FiNUFFTError();
}

template <>
void Nufft3d3<float>::execute(const std::complex<float>* cj, std::complex<float>* fk) {
  if (finufftf_execute(*reinterpret_cast<const finufftf_plan*>(plan_.get()),
                       const_cast<std::complex<float>*>(cj), fk))
    throw FiNUFFTError();
}

}  // namespace host
}  // namespace bipp
