#include "bipp/standard_synthesis.hpp"

#include <chrono>
#include <complex>
#include <optional>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "context_internal.hpp"
#include "memory/view.hpp"
#include "imager.hpp"

#ifdef BIPP_MPI
#include "communicator_internal.hpp"
#endif

namespace bipp {

template <typename T>
StandardSynthesis<T>::StandardSynthesis(Context& ctx, StandardSynthesisOptions opt,
                                        std::size_t nLevel, std::size_t nPixel, const T* pixelX,
                                        const T* pixelY, const T* pixelZ) {
  try {
#ifdef BIPP_MPI
    if (ctx.communicator().size() > 1) {
      const auto& commInt = InternalCommunicatorAccessor::get(ctx.communicator());
      plan_ = decltype(plan_)(
          new Imager<T>(Imager<T>::distributed_standard_synthesis(
              commInt, InternalContextAccessor::get(ctx), opt, nLevel,
              ConstView<T, 1>(pixelX, nPixel, 1), ConstView<T, 1>(pixelY, nPixel, 1),
              ConstView<T, 1>(pixelZ, nPixel, 1))),
          [](auto&& ptr) { delete reinterpret_cast<Imager<T>*>(ptr); });
    } else
#endif
      plan_ = decltype(plan_)(
          new Imager<T>(Imager<T>::standard_synthesis(
              InternalContextAccessor::get(ctx), opt, nLevel, ConstView<T, 1>(pixelX, nPixel, 1),
              ConstView<T, 1>(pixelY, nPixel, 1), ConstView<T, 1>(pixelZ, nPixel, 1))),
          [](auto&& ptr) { delete reinterpret_cast<Imager<T>*>(ptr); });
  } catch (const std::exception& e) {
    try {
      InternalContextAccessor::get(ctx)->logger().log(
          BIPP_LOG_LEVEL_ERROR, "StandardSynthesis creation error: {}", e.what());
    } catch (...) {
    }
    throw;
  }
}

template <typename T>
auto StandardSynthesis<T>::collect(
    std::size_t nAntenna, std::size_t nBeam, T wl,
    const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc, const std::complex<T>* s,
    std::size_t lds, const std::complex<T>* w, std::size_t ldw, const T* xyz, std::size_t ldxyz)
    -> void {
  try {
    reinterpret_cast<Imager<T>*>(plan_.get())
        ->collect(wl, eigMaskFunc, ConstView<std::complex<T>, 2>(s, {nBeam, nBeam}, {1, lds}),
                  ConstView<std::complex<T>, 2>(w, {nAntenna, nBeam}, {1, ldw}),
                  ConstView<T, 2>(xyz, {nAntenna, 3}, {1, ldxyz}), ConstView<T, 2>());
  } catch (const std::exception& e) {
    try {
      reinterpret_cast<Imager<T>*>(plan_.get())
          ->context()
          .logger()
          .log(BIPP_LOG_LEVEL_ERROR, "StandardSynthesis.collect() error: {}", e.what());
    } catch (...) {
    }
    throw;
  }
}

template <typename T>
auto StandardSynthesis<T>::get(T* out, std::size_t ld) -> void {
  try {
    reinterpret_cast<Imager<T>*>(plan_.get())->get(out, ld);
  } catch (const std::exception& e) {
    try {
      reinterpret_cast<Imager<T>*>(plan_.get())
          ->context()
          .logger()
          .log(BIPP_LOG_LEVEL_ERROR, "StandardSynthesis.get() error: {}", e.what());
    } catch (...) {
    }
    throw;
  }
}

template class BIPP_EXPORT StandardSynthesis<double>;

template class BIPP_EXPORT StandardSynthesis<float>;

extern "C" {
BIPP_EXPORT BippError bipp_standard_synthesis_create_f(BippContext ctx,
                                                       BippNufftSynthesisOptions opt, size_t nLevel,
                                                       size_t nPixel, const float* lmnX,
                                                       const float* lmnY, const float* lmnZ,
                                                       BippStandardSynthesisF* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new StandardSynthesis<float>(*reinterpret_cast<Context*>(ctx),
                                         *reinterpret_cast<StandardSynthesisOptions*>(opt), nLevel,
                                         nPixel, lmnX, lmnY, lmnZ);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_destroy_f(BippStandardSynthesisF* plan) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<StandardSynthesis<float>*>(*plan);
    *plan = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_collect_f(BippStandardSynthesisF plan,
                                                        size_t nAntenna, size_t nBeam, float wl,
                                                        void (*mask)(size_t, size_t, float*),
                                                        const void* s, size_t lds, const void* w,
                                                        size_t ldw, const float* xyz,
                                                        size_t ldxyz) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<float>*>(plan)->collect(
        nAntenna, nBeam, wl, mask, reinterpret_cast<const std::complex<float>*>(s), lds,
        reinterpret_cast<const std::complex<float>*>(w), ldw, xyz, ldxyz);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_get_f(BippStandardSynthesisF plan, float* img,
                                                    size_t ld) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<float>*>(plan)->get(img, ld);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_create(BippContext ctx, BippStandardSynthesisOptions opt, size_t nLevel,
                                                     size_t nPixel, const double* lmnX,
                                                     const double* lmnY, const double* lmnZ,
                                                     BippStandardSynthesis* plan) {
  if (!ctx) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    *plan = new StandardSynthesis<double>(*reinterpret_cast<Context*>(ctx),
                                          *reinterpret_cast<StandardSynthesisOptions*>(opt), nLevel,
                                          nPixel, lmnX, lmnY, lmnZ);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_destroy(BippStandardSynthesis* plan) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<StandardSynthesis<double>*>(*plan);
    *plan = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_collect(BippStandardSynthesis plan, size_t nAntenna,
                                                      size_t nBeam, double wl,
                                                      void (*mask)(size_t, size_t, double*),
                                                      const void* s, size_t lds, const void* w,
                                                      size_t ldw, const double* xyz, size_t ldxyz) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<double>*>(plan)->collect(
        nAntenna, nBeam, wl, mask, reinterpret_cast<const std::complex<double>*>(s), lds,
        reinterpret_cast<const std::complex<double>*>(w), ldw, xyz, ldxyz);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_standard_synthesis_get(BippStandardSynthesis plan, double* img,
                                                  size_t ld) {
  if (!plan) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesis<double>*>(plan)->get(img, ld);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ss_options_create(BippStandardSynthesisOptions* opt) {
  try {
    *reinterpret_cast<StandardSynthesisOptions**>(opt) = new StandardSynthesisOptions();
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ss_options_destroy(BippStandardSynthesisOptions* opt) {
  if (!opt || !(*opt)) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    delete *reinterpret_cast<StandardSynthesisOptions**>(opt);
    *reinterpret_cast<Context**>(opt) = nullptr;
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

BIPP_EXPORT BippError bipp_ss_options_set_collect_group_size(BippStandardSynthesisOptions opt,
                                                             size_t size) {
  if (!opt) {
    return BIPP_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<StandardSynthesisOptions*>(opt)->set_collect_group_size(size);
  } catch (const bipp::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return BIPP_UNKNOWN_ERROR;
  }
  return BIPP_SUCCESS;
}

}

}  // namespace bipp
