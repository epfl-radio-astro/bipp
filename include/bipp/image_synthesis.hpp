#pragma once

#include <bipp/config.h>
//---
#include <bipp/enums.h>

#include <array>
#include <bipp/context.hpp>
#include <bipp/dataset.hpp>
#include <bipp/image_prop.hpp>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

/*! \cond PRIVATE */
namespace bipp {
/*! \endcond */

struct Partition {
  Partition() = default;

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

  Partition(Partition::Grid g) : method(std::move(g)) {}

  Partition(Partition::Auto g) : method(std::move(g)) {}

  Partition(Partition::None g) : method(std::move(g)) {}

  std::variant<Partition::Auto, Partition::None, Partition::Grid> method = Partition::Auto();
};

struct NufftSynthesisOptions {
  /**
   * Floating point pricision to use internally.
   */
  BippPrecision precision = BIPP_PRECISION_SINGLE;

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
   * The partition method used in the image domain. Partitioning decreases memory usage, but may
   * come with a performance penalty.
   */
  Partition localUVWPartition = Partition{Partition::Auto()};

  /**
   * Normalize image by the number of collect steps.
   */
  bool normalizeImage = true;

  /**
   * Apply scaling defined in dataset.
   */
  bool apply_scaling = true;

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
   * Set scaling setting.
   *
   * @param[in] apply True or false.
   */
  inline auto set_apply_scaling(bool apply) -> NufftSynthesisOptions& {
    apply_scaling = apply;
    return *this;
  }

  /**
   * Set floating point precision.
   *
   * @param[in] prec Precision.
   */
  inline auto set_precision(BippPrecision prec) -> NufftSynthesisOptions& {
    precision = prec;
    return *this;
  }
};

struct StandardSynthesisOptions {
  /**
   * Floating point pricision to use internally.
   */
  BippPrecision precision = BIPP_PRECISION_SINGLE;

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
  bool apply_scaling = true;

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
   * Set scaling setting.
   *
   * @param[in] apply True or false.
   */
  inline auto set_apply_scaling(bool apply) -> StandardSynthesisOptions& {
    apply_scaling = apply;
    return *this;
  }

  /**
   * Set floating point precision.
   *
   * @param[in] prec Precision.
   */
  inline auto set_precision(BippPrecision prec) -> StandardSynthesisOptions& {
    precision = prec;
    return *this;
  }
};

//TODO: doc
void image_synthesis(
    Context& ctx, const std::variant<NufftSynthesisOptions, StandardSynthesisOptions>& opt,
    Dataset& dataset,
    std::unordered_map<std::string, std::vector<std::pair<std::size_t, const float*>>> selection,
    ImageProp& imageProp, const std::string& imageFileName);

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
