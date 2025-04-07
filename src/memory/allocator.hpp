#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"

namespace bipp {

enum class MemoryType { Host, Device };

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

  virtual auto size() -> std::optional<std::uint_least64_t> { return std::nullopt; }

  virtual auto type() -> MemoryType = 0;
};
}  // namespace bipp
