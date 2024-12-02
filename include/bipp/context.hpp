#pragma once

#include <bipp/config.h>
#include <bipp/enums.h>

#include <bipp/communicator.hpp>
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
   * Constructor of a distributed Context with default configuration for given processing
   * unit.
   *
   * @param[in] pu Processing unit to be used for computations.
   * @param[in] comm Communicator to use for distributed image synthesis.
   */
  Context(BippProcessingUnit pu, Communicator comm);

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

  /**
   * Access a Context parameter.
   * @return Processing unit used.
   */
  auto communicator() const -> const Communicator&;

  /**
   * Attach to root process to enable distributed image synthesis.
   * Must be called by all non-root processes. May be called by root process, in which case it will
   * return false immediately. Non-root processes are only detached once the communicator and
   * context of the root process are destroyed.
   *
   * @return True if process was attached, false otherwise.
   */
  auto attach_non_root() -> bool;

private:
  /*! \cond PRIVATE */
  friend InternalContextAccessor;

  std::shared_ptr<ContextInternal> ctx_;

  Communicator comm_;
  /*! \endcond */
};

/*! \cond PRIVATE */
}  // namespace bipp
/*! \endcond */
