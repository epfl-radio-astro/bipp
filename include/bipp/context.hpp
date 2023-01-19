#pragma once

#include <bipp/bipp.h>
#include <bipp/config.h>

#include <memory>

/*! \cond PRIVATE */
namespace bipp {

class ContextInternal;

struct InternalContextAccessor;
/*! \endcond */

class BIPP_EXPORT Context {
public:
  /**
   * Constructor of Context with default configuration for given processing
   * unit.
   *
   * @param[in] pu Processing unit to be used for computations.
   */
  explicit Context(BippProcessingUnit pu);

  /**
   * Default move constructor.
   */
  Context(Context&&) = default;

  /**
   * Disabled copy constructor.
   */
  Context(const Context&) = delete;

  /**
   * Default move assignment operator.
   */
  Context& operator=(Context&&) = default;

  /**
   * Disabled copy assignment operator.
   */
  Context& operator=(const Context&) = delete;

  /**
   * Access a Context parameter.
   * @return Processing unit used.
   */
  auto processing_unit() const -> BippProcessingUnit;

private:
  /*! \cond PRIVATE */
  friend InternalContextAccessor;

  std::shared_ptr<ContextInternal> ctx_;
  /*! \endcond */
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
