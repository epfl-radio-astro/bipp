#pragma once

#include <cstddef>
#include <cstdint>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"

namespace bipp {

class Allocator {
public:
  Allocator() = default;

  Allocator(const Allocator&) = delete;

  Allocator(Allocator&&) = default;

  auto operator=(const Allocator&) -> Allocator& = delete;

  auto operator=(Allocator&&) -> Allocator& = default;

  virtual ~Allocator() = default;

  virtual auto allocate(std::size_t size) -> void* = 0;

  virtual auto deallocate(void* ptr) -> void = 0;

  virtual auto size() -> std::uint_least64_t = 0;
};
}  // namespace bipp
