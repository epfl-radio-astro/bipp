#include <cstddef>

#include "bipp/config.h"
#include "gpu/util/cub_api.hpp"
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

template <typename T>
auto min_element(Queue& q, std::size_t n, const T* x, T* minElement) -> void {
  std::size_t worksize = 0;
  api::check_status(api::cub::DeviceReduce::Min<const T*, T*>(nullptr, worksize, nullptr, nullptr,
                                                              n, q.stream()));

  auto workBuffer = q.create_device_buffer<char>(worksize);

  api::check_status(api::cub::DeviceReduce::Min<const T*, T*>(workBuffer.get(), worksize, x,
                                                              minElement, n, q.stream()));
}


template <typename T>
auto max_element(Queue& q, std::size_t n, const T* x, T* maxElement) -> void {
  std::size_t worksize = 0;
  api::check_status(api::cub::DeviceReduce::Max<const T*, T*>(nullptr, worksize, nullptr, nullptr,
                                                              n, q.stream()));

  auto workBuffer = q.create_device_buffer<char>(worksize);

  api::check_status(api::cub::DeviceReduce::Max<const T*, T*>(workBuffer.get(), worksize, x,
                                                              maxElement, n, q.stream()));
}

template auto min_element<float>(Queue& q, std::size_t n, const float* x, float* minElement)
    -> void;

template auto min_element<double>(Queue& q, std::size_t n, const double* x, double* minElement)
    -> void;

template auto max_element<float>(Queue& q, std::size_t n, const float* x, float* maxElement)
    -> void;

template auto max_element<double>(Queue& q, std::size_t n, const double* x, double* maxElement)
    -> void;

}  // namespace gpu
}  // namespace bipp
