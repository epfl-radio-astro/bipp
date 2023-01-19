#pragma once

#include <complex>
#include <functional>
#include <memory>

#include "bipp/config.h"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
class Nufft3d3 {
public:
  using planType = std::unique_ptr<void, std::function<void(void*)>>;

  Nufft3d3(int iflag, T tol, std::size_t numTrans, std::size_t M, const T* x, const T* y,
           const T* z, std::size_t N, const T* s, const T* t, const T* u);

  void execute(const api::ComplexType<T>* cj, api::ComplexType<T>* fk);

private:
  planType plan_;
};

}  // namespace gpu
}  // namespace bipp
