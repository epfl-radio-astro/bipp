#include "bipp/nufft_synthesis.hpp"

#include <chrono>
#include <complex>
#include <optional>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "imager.hpp"

#ifdef BIPP_MPI
#include "communicator_internal.hpp"
#endif

namespace bipp {
template <typename T>
NufftSynthesis<T>::NufftSynthesis(Context& ctx, NufftSynthesisOptions opt, std::size_t nImages,
                                  std::size_t nPixel, const T* lmnX, const T* lmnY, const T* lmnZ) {
  try {
    plan_ = decltype(plan_)(new Imager<T>(Imager<T>::nufft_synthesis(
                                InternalContextAccessor::get(ctx), std::move(opt), nImages,
                                ConstView<T, 1>(lmnX, nPixel, 1), ConstView<T, 1>(lmnY, nPixel, 1),
                                ConstView<T, 1>(lmnZ, nPixel, 1))),
                            [](auto&& ptr) { delete reinterpret_cast<Imager<T>*>(ptr); });
  } catch (const std::exception& e) {
    try {
      InternalContextAccessor::get(ctx)->logger().log(
          BIPP_LOG_LEVEL_ERROR, "NufftSynthesis creation error: {}", e.what());
    } catch (...) {
    }
    throw;
  }
}

#ifdef BIPP_MPI
template <typename T>
NufftSynthesis<T>::NufftSynthesis(Communicator& comm, Context& ctx, NufftSynthesisOptions opt,
                                  std::size_t nImages, std::size_t nPixel, const T* lmnX,
                                  const T* lmnY, const T* lmnZ) {
  try {
    const auto& commInt =  InternalCommunicatorAccessor::get(comm);
    if(commInt->comm().size() <= 1) {
      plan_ =
          decltype(plan_)(new Imager<T>(Imager<T>::nufft_synthesis(
                              InternalContextAccessor::get(ctx), std::move(opt), nImages,
                              ConstView<T, 1>(lmnX, nPixel, 1), ConstView<T, 1>(lmnY, nPixel, 1),
                              ConstView<T, 1>(lmnZ, nPixel, 1))),
                          [](auto&& ptr) { delete reinterpret_cast<Imager<T>*>(ptr); });
    } else {
      plan_ =
          decltype(plan_)(new Imager<T>(Imager<T>::distributed_nufft_synthesis(
                              commInt, InternalContextAccessor::get(ctx), std::move(opt), nImages,
                              ConstView<T, 1>(lmnX, nPixel, 1), ConstView<T, 1>(lmnY, nPixel, 1),
                              ConstView<T, 1>(lmnZ, nPixel, 1))),
                          [](auto&& ptr) { delete reinterpret_cast<Imager<T>*>(ptr); });
    }
  } catch (const std::exception& e) {
    try {
      InternalContextAccessor::get(ctx)->logger().log(
          BIPP_LOG_LEVEL_ERROR, "NufftSynthesis creation error: {}", e.what());
    } catch (...) {
    }
    throw;
  }
}
#endif

template <typename T>
auto NufftSynthesis<T>::collect(
    std::size_t nAntenna, std::size_t nBeam, T wl,
    const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc, const std::complex<T>* s,
    std::size_t lds, const std::complex<T>* w, std::size_t ldw, const T* xyz, std::size_t ldxyz,
    const T* uvw, std::size_t lduvw) -> void {
  try {
    reinterpret_cast<Imager<T>*>(plan_.get())
        ->collect(wl, eigMaskFunc, ConstView<std::complex<T>, 2>(s, {nBeam, nBeam}, {1, lds}),
                  ConstView<std::complex<T>, 2>(w, {nAntenna, nBeam}, {1, ldw}),
                  ConstView<T, 2>(xyz, {nAntenna, 3}, {1, ldxyz}),
                  ConstView<T, 2>(uvw, {nAntenna * nAntenna, 3}, {1, lduvw}));
  } catch (const std::exception& e) {
    try {
      reinterpret_cast<Imager<T>*>(plan_.get())
          ->context()
          .logger()
          .log(BIPP_LOG_LEVEL_ERROR, "NufftSynthesis.collect() error: {}", e.what());
    } catch (...) {
    }
    throw;
  }
}

template <typename T>
auto NufftSynthesis<T>::get(T* out, std::size_t ld) -> void {
  try {
    reinterpret_cast<Imager<T>*>(plan_.get())->get(out, ld);
  } catch (const std::exception& e) {
    try {
      reinterpret_cast<Imager<T>*>(plan_.get())
          ->context()
          .logger()
          .log(BIPP_LOG_LEVEL_ERROR, "NufftSynthesis.get() error: {}", e.what());
    } catch (...) {
    }
    throw;
  }
}

template class BIPP_EXPORT NufftSynthesis<double>;

template class BIPP_EXPORT NufftSynthesis<float>;

extern "C" {
BIPP_EXPORT BippError bipp_ns_options_create(BippNufftSynthesisOptions* opt) {
  try {
    *reinterpret_cast<NufftSynthesisOptions**>(opt) = new NufftSynthesisOptions();
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_destroy(BippNufftSynthesisOptions* opt) {
  if (!opt || !(*opt)) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete *reinterpret_cast<NufftSynthesisOptions**>(opt);
    *reinterpret_cast<Context**>(opt) = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_tolerance(BippNufftSynthesisOptions opt, float tol) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_tolerance(tol);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_collect_group_size(BippNufftSynthesisOptions opt,
                                                             size_t size) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_collect_group_size(size);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError
bipp_ns_options_set_local_image_partition_auto(BippNufftSynthesisOptions opt) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_image_partition({Partition::Auto{}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError
bipp_ns_options_set_local_image_partition_none(BippNufftSynthesisOptions opt) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_image_partition({Partition::None{}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_local_image_partition_grid(BippNufftSynthesisOptions opt,
                                                                     size_t dimX, size_t dimY,
                                                                     size_t dimZ) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_image_partition(
        {Partition::Grid{{dimX, dimY, dimZ}}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_local_uvw_partition_auto(BippNufftSynthesisOptions opt) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_uvw_partition({Partition::Auto{}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_local_uvw_partition_none(BippNufftSynthesisOptions opt) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_uvw_partition({Partition::None{}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ns_options_set_local_uvw_partition_grid(BippNufftSynthesisOptions opt,
                                                                   size_t dimX, size_t dimY,
                                                                   size_t dimZ) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesisOptions*>(opt)->set_local_uvw_partition(
        {Partition::Grid{{dimX, dimY, dimZ}}});
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_create_f(BippContext ctx, BippNufftSynthesisOptions opt,
                                                    size_t nImages, size_t nPixel,
                                                    const float* lmnX, const float* lmnY,
                                                    const float* lmnZ, BippNufftSynthesisF* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new NufftSynthesis<float>(*reinterpret_cast<Context*>(ctx),
                                      *reinterpret_cast<const NufftSynthesisOptions*>(opt), nImages,
                                      nPixel, lmnX, lmnY, lmnZ);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_destroy_f(BippNufftSynthesisF* plan) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<NufftSynthesis<float>*>(*plan);
    *plan = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_collect_f(BippNufftSynthesisF plan, size_t nAntenna,
                                                     size_t nBeam, float wl,
                                                     void (*mask)(size_t, size_t, float*),
                                                     const void* s, size_t lds, const void* w,
                                                     size_t ldw, const float* xyz, size_t ldxyz,
                                                     const float* uvw, size_t lduvw) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesis<float>*>(plan)->collect(
        nAntenna, nBeam, wl, mask, reinterpret_cast<const std::complex<float>*>(s), lds,
        reinterpret_cast<const std::complex<float>*>(w), ldw, xyz, ldxyz, uvw, lduvw);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_get_f(BippNufftSynthesisF plan, float* img, size_t ld) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesis<float>*>(plan)->get(img, ld);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_create(BippContext ctx, BippNufftSynthesisOptions opt,
                                                 size_t nImages,
                                                  size_t nPixel, const double* lmnX,
                                                  const double* lmnY, const double* lmnZ,
                                                  BippNufftSynthesis* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new NufftSynthesis<double>(*reinterpret_cast<Context*>(ctx),
                                       *reinterpret_cast<const NufftSynthesisOptions*>(opt),
                                       nImages, nPixel, lmnX, lmnY, lmnZ);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_destroy(BippNufftSynthesis* plan) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<NufftSynthesis<double>*>(*plan);
    *plan = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_collect(BippNufftSynthesis plan, size_t nAntenna,
                                                   size_t nBeam, double wl,
                                                   void (*mask)(size_t, size_t, double*),
                                                   const void* s, size_t lds, const void* w,
                                                   size_t ldw, const double* xyz, size_t ldxyz,
                                                   const double* uvw, size_t lduvw) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesis<double>*>(plan)->collect(
        nAntenna, nBeam, wl, mask, reinterpret_cast<const std::complex<double>*>(s), lds,
        reinterpret_cast<const std::complex<double>*>(w), ldw, xyz, ldxyz, uvw, lduvw);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_nufft_synthesis_get(BippNufftSynthesis plan, double* img,
                                               size_t ld) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<NufftSynthesis<double>*>(plan)->get(img, ld);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}
}

}  // namespace bipp
