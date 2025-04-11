#include "memory/allocator_factory.hpp"

#include <cstdlib>
#include <memory>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "memory/allocator.hpp"
#include "memory/pool_allocator.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/runtime_api.hpp"
#endif

#ifdef BIPP_UMPIRE
#include "memory/umpire_allocator.hpp"
#endif

namespace bipp {

auto AllocatorFactory::simple_host() -> std::unique_ptr<Allocator> {
  class SimpleAllocator : public Allocator {
  public:
    auto allocate(std::size_t size) -> void* override {
      void* ptr = nullptr;
      if (size) {
        ptr = std::malloc(size);
        if (!ptr) throw std::bad_alloc();
      }
      return ptr;
    }

    auto deallocate(void* ptr) -> void override {
      if (ptr) std::free(ptr);
    }

    auto type() -> MemoryType override { return MemoryType::Host; }
  };

  return std::make_unique<SimpleAllocator>();
}

auto AllocatorFactory::host() -> std::unique_ptr<Allocator> {
#ifdef BIPP_UMPIRE
  return std::unique_ptr<Allocator>{MemoryType::Host, new UmpireAllocator("HOST")};
#else
  return std::unique_ptr<Allocator>{new PoolAllocator(
      MemoryType::Host,
      [](std::size_t size) -> void* {
        void* ptr = nullptr;
        if (size) {
          ptr = std::malloc(size);
          if (!ptr) throw std::bad_alloc();
        }
        return ptr;
      },
      [](void* ptr) -> void {
        if (ptr) std::free(ptr);
      })};
#endif
}

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)

auto AllocatorFactory::pinned() -> std::unique_ptr<Allocator> {
#ifdef BIPP_UMPIRE
  return std::unique_ptr<Allocator>{MemoryType::Host, new UmpireAllocator("PINNED")};
#else
  return std::unique_ptr<Allocator>{new PoolAllocator(
      MemoryType::Host,
      [](std::size_t size) -> void* {
        void* ptr = nullptr;
        if (size) gpu::api::malloc_host(&ptr, size);
        return ptr;
      },
      [](void* ptr) -> void {
        if (ptr) std::ignore = gpu::api::free_host(ptr);
      })};
#endif
}

auto AllocatorFactory::device() -> std::unique_ptr<Allocator> {
#ifdef BIPP_UMPIRE
  return std::unique_ptr<Allocator>{MemoryType::Device, new UmpireAllocator("DEVICE")};
#else
  return std::unique_ptr<Allocator>{new PoolAllocator(
      MemoryType::Device,
      [](std::size_t size) -> void* {
        void* ptr = nullptr;
        if (size) gpu::api::malloc(&ptr, size);
        return ptr;
      },
      [](void* ptr) -> void {
        if (ptr) std::ignore = gpu::api::free(ptr);
      })};
#endif
}

#endif
}  // namespace bipp
