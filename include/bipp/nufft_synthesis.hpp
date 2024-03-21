#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>
#ifdef BIPP_MPI
#include <bipp/communicator.hpp>
#endif

#include <array>
#include <bipp/context.hpp>
#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

struct Partition {
  /**
   * Automatic domain partition method setting.
   */
  struct Auto {};

  /**
   * Disable domain partitioning.
   */
  struct None {};

  /**
   * Use regular grid domain partitioning
   */
  struct Grid {
    /**
     * Grid dimension.
     */
    std::array<std::size_t, 3> dimensions = {1, 1, 1};
  };

  std::variant<Partition::Auto, Partition::None, Partition::Grid> method = Partition::Auto();
};

struct NufftSynthesisOptions {
  /**
   * The tolerance used when computing the NUFFT. Smaller value will increase accuracy but requires
   * more operations.
   */
  float tolerance = 0.001f;

  /**
   * The maximum number of collected datasets processed together. Larger number typically improves
   * performance but requires more memory. Internal heuristic is used if unset.
   */
  std::optional<std::size_t> collectGroupSize = std::nullopt;

  /**
   * The partition method used in the UVW domain. Partitioning decreases memory usage, but may come
   * with a performance penalty.
   */
  Partition localImagePartition = Partition{Partition::Auto()};

  /**
   * The partition method used in the image domain. Partitioning decreases memory usage, but may
   * come with a performance penalty.
   */
  Partition localUVWPartition = Partition{Partition::Auto()};

  /**
   * Normalize image by the number of collect steps.
   */
  bool normalizeImage = true;

  /**
   * Normalize image by the number of non-zero visibilities.
   */
  bool normalizeImageNvis = true;

  /**
   * Set the tolerance.
   *
   * @param[in] tol Tolerance.
   */
  inline auto set_tolerance(float tol) -> NufftSynthesisOptions& {
    tolerance = tol;
    return *this;
  }

  /**
   * Set the collection group size.
   *
   * @param[in] size Collection group size.
   */
  inline auto set_collect_group_size(std::optional<std::size_t> size) -> NufftSynthesisOptions& {
    collectGroupSize = size;
    return *this;
  }

  /**
   * Set the partitioning method for the UVW domain.
   *
   * @param[in] p Partition method.
   */
  inline auto set_local_image_partition(Partition p) -> NufftSynthesisOptions& {
    localImagePartition = std::move(p);
    return *this;
  }

  /**
   * Set the partitioning method for the image domain.
   *
   * @param[in] p Partition method.
   */
  inline auto set_local_uvw_partition(Partition p) -> NufftSynthesisOptions& {
    localUVWPartition = std::move(p);
    return *this;
  }

  /**
   * Set normalization of image.
   *
   * @param[in] normalize True or false.
   */
  inline auto set_normalize_image(bool normalize) -> NufftSynthesisOptions& {
    normalizeImage = normalize;
    return *this;
  }

  /**
   * Set normalization of image by number of non-zero visibilities.
   *
   * @param[in] normalize True or false.
   */
  inline auto set_normalize_image_by_nvis(bool normalize) -> NufftSynthesisOptions& {
    normalizeImageNvis = normalize;
    return *this;
  }
};

template <typename T>
class BIPP_EXPORT NufftSynthesis {
public:
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
  using valueType = T;

  /**
   * Create a nufft synthesis plan.
   *
   * @param[in] ctx Context handle.
   * @param[in] opt Options.
   * @param[in] nImagers Number of images.
   * @param[in] nPixel Number of image pixels.
   * @param[in] lmnX Array of image x coordinates of size nPixel.
   * @param[in] lmnY Array of image y coordinates of size nPixel.
   * @param[in] lmnZ Array of image z coordinates of size nPixel.
   */
  NufftSynthesis(Context& ctx, NufftSynthesisOptions opt, std::size_t nImagers,
                 std::size_t nPixel, const T* lmnX,
                 const T* lmnY, const T* lmnZ);


  /**
   * Collect radio data.
   *
   * @param[in] nAntenna Number of antenna.
   * @param[in] nBeam Number of beam.
   * @param[in] wl The wavelength.
   * @param[in] eigMaskFunc Function, that allows mutable access to the computed eigenvalues. Will
   * be called with the level index, number of eigenvalues and a pointer to the eigenvalue array.
   * @param[in] s Optional complex 2D sensitivity array of size (nBeam, nBeam). May be null.
   * @param[in] lds Leading dimension of s.
   * @param[in] w 2D complex beamforming array of size (nAntenna, nBeam).
   * @param[in] ldw Leading dimension of w.
   * @param[in] xyz 2D antenna position array of size (nAntenna, 3).
   * @param[in] ldxyz Leading dimension of xyz.
   * @param[in] uvw UVW coordinates expressed in the local UVW frame of size (nAntenna * nAntenna,
   * 3).
   * @param[in] lduvw Leading dimension of uvw.
   */
  auto collect(std::size_t nAntenna, std::size_t nBeam, T wl,
               const std::function<void(std::size_t, std::size_t, T*)>& eigMaskFunc,
               const std::complex<T>* s, std::size_t lds, const std::complex<T>* w, std::size_t ldw,
               const T* xyz, std::size_t ldxyz, const T* uvw, std::size_t lduvw) -> void;

  /**
   * Get image.
   *
   * @param[out] img 2D image array of size (nPixel, nImagers).
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
