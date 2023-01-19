#pragma once

#include <cstddef>
#include <memory>

#include "bipp/config.h"
#include "memory/allocator.hpp"

namespace bipp {

template <typename T>
class Buffer {
public:
  Buffer() : size_(0), data_() {}

  Buffer(const std::shared_ptr<Allocator>& alloc, std::size_t size) : size_(size) {
    if (size) {
      auto ptr = alloc->allocate(size * sizeof(T));
      data_ = std::shared_ptr<void>(ptr, [=](void* p) {
        // implicit alloc copy
        if (p) alloc->deallocate(p);
      });
    }
  }

  Buffer(const Buffer&) = delete;

  Buffer(Buffer&& b) { *this = std::move(b); }

  auto operator=(const Buffer&) -> Buffer& = delete;

  auto operator=(Buffer&& b) -> Buffer& {
    data_ = std::move(b.data_);
    size_ = b.size_;
    b.size_ = 0;
    return *this;
  };

  explicit operator bool() const noexcept { return size_; }

  inline auto get() -> T* { return reinterpret_cast<T*>(data_.get()); }

  inline auto get() const -> const T* { return reinterpret_cast<T*>(data_.get()); }

  inline auto size() const noexcept -> std::size_t { return size_; }

  inline auto size_in_bytes() const noexcept -> std::size_t { return size_ * sizeof(T); }

  inline auto data_handler() const noexcept -> const std::shared_ptr<void>& { return data_; }

private:
  std::size_t size_;
  std::shared_ptr<void> data_;
};

}  // namespace bipp
