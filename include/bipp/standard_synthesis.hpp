#pragma once

#include <bipp/config.h>
#ifdef BIPP_MPI
#include <bipp/communicator.hpp>
#endif

#include <bipp/context.hpp>
#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */
struct StandardSynthesisOptions {
  /**
   * The maximum number of collected datasets processed together. Only benefits distributed image
   * synthesis.
   */
  std::optional<std::size_t> collectGroupSize = std::nullopt;

  /**
   * Normalize image by the number of collect steps.
   */
  bool normalizeImage = true;

  /**
   * Normalize image by the number of non-zero visibilities.
   */
  bool normalizeImageNvis = true;

  /**
   * Set the collection group size.
   *
   * @param[in] size Collection group size.
   */
  inline auto set_collect_group_size(std::optional<std::size_t> size) -> StandardSynthesisOptions& {
    collectGroupSize = size;
    return *this;
  }

  /**
   * Set normalization of image.
   *
   * @param[in] normalize True or false.
   */
  inline auto set_normalize_image(bool normalize) -> StandardSynthesisOptions& {
    normalizeImage = normalize;
    return *this;
  }

  /**
   * Set normalization of image by number of non-zero visibilities.
   *
   * @param[in] normalize True or false.
   */
  inline auto set_normalize_image_by_nvis(bool normalize) -> StandardSynthesisOptions& {
    normalizeImageNvis = normalize;
    return *this;
  }
};


template <typename T>
class BIPP_EXPORT StandardSynthesis {
public:
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
  using valueType = T;

  /**
   * Create a standard synthesis plan.
   *
   * @param[in] ctx Context handle.
   * @param[in] opt Options.
   * @param[in] nImages Number of images.
   * @param[in] nPixel Number of image pixels.
   * @param[in] lmnX Array of image x coordinates of size nPixel.
   * @param[in] lmnY Array of image y coordinates of size nPixel.
   * @param[in] lmnZ Array of image z coordinates of size nPixel.
   */
  StandardSynthesis(Context& ctx, StandardSynthesisOptions opt, std::size_t nImages,
                    std::size_t nPixel, const T* lmnX, const T* lmnY, const T* lmnZ);

  /**
   * Collect radio data.
   *
   * @param[in] nAntenna Number of antenna.
   * @param[in] nBeam Number of beam.
   * @param[in] wl The wavelength.
   * @param[in] eigMaskFunc Function, that allows mutable access to the computed eigenvalues. Will
   * be called with the level index, number of eigenvalues and a pointer to the eigenvalue array.
   * @param[in] s Optional 2D sensitivity array of size (nBeam, nBeam). May be null.
   * @param[in] lds Leading dimension of s.
   * @param[in] w 2D beamforming array of size (nAntenna, nBeam).
   * @param[in] ldw Leading dimension of w.
   * @param[in] xyz 2D antenna position array of size (nAntenna, 3).
   * @param[in] ldxyz Leading dimension of xyz.
   */
  auto collect(std::size_t nAntenna, std::size_t nBeam, T wl,
               const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc,
               const std::complex<T>* s, std::size_t lds, const std::complex<T>* w, std::size_t ldw,
               const T* xyz, std::size_t ldxyz) -> void;

  /**
   * Get image.
   *
   * @param[out] img 2D image array of size (nPixel, nImages).
   * @param[in] ld Leading dimension of img.
   */
  auto get(T* img, std::size_t ld) -> void;

private:
  /*! \cond PRIVATE */
  std::unique_ptr<void, std::function<void(void*)>> plan_;
  /*! \endcond */
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
