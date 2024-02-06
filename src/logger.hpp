#pragma once

#include <any>
#include <complex>
#include <cstddef>
#include <memory>
#include <string>
#include <sstream>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "memory/view.hpp"
#include "rt_graph.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/runtime_api.hpp"
#endif

#include <spdlog/logger.h>

namespace bipp {

// helper function to print object adresses
template <typename T>
inline auto pointer_to_string(const T* ptr) -> std::string {
  std::stringstream s;
  s << ptr;
  return s.str();
}

class Logger {
public:
   explicit Logger(BippLogLevel level, const char* out = "stdout");

   Logger(const Logger&) = delete;

   Logger(Logger&&) = default;

   auto operator=(const Logger&) -> Logger& = delete;

   auto operator=(Logger&&) -> Logger& = default;

   template <typename... Args>
   auto log(BippLogLevel level, const std::string_view& s, Args&&... args) -> void {
     if (level <= level_) logger_->log(convert_level(level), s, std::forward<Args>(args)...);
   }

   // Internally store timings. Does not print to log until log_timings() is called
   template <std::size_t N>
   auto scoped_timing(BippLogLevel level, const char (&tag)[N]) -> rt_graph::ScopedTiming {
     if (level <= level_) {
       return rt_graph::ScopedTiming(tag, timer_);
     }
     return rt_graph::ScopedTiming();
   }

   auto scoped_timing(BippLogLevel level, std::string tag) -> rt_graph::ScopedTiming {
     if (level <= level_) {
       return rt_graph::ScopedTiming(std::move(tag), timer_);
     }
     return rt_graph::ScopedTiming();
   }

   auto log_timings(BippLogLevel level) -> void;

   // log 2D arrays
   auto log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m, std::size_t n,
                   const float* array, std::size_t ld) -> void;
   auto log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m, std::size_t n,
                   const double* array, std::size_t ld) -> void;
   auto log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m, std::size_t n,
                   const std::complex<float>* array, std::size_t ld) -> void;
   auto log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m, std::size_t n,
                   const std::complex<double>* array, std::size_t ld) -> void;

   template <typename T>
   inline auto log_matrix(BippLogLevel level, const std::string_view& s, ConstView<T, 2> array) {
     this->log_matrix(level, s, array.shape(0), array.shape(1), array.data(),
                      array.strides(1));
   }

   template <typename T>
   inline auto log_matrix(BippLogLevel level, const std::string_view& s, ConstView<T, 1> array) {
     this->log_matrix(level, s, array.shape(), 1, array.data(), array.shape());
   }

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
   inline auto log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m, std::size_t n,
                   const gpu::api::ComplexType<float>* array, std::size_t ld) -> void {
     log_matrix(level, s, m, n, reinterpret_cast<const std::complex<float>*>(array), ld);
   }
   inline auto log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m, std::size_t n,
                   const gpu::api::ComplexType<double>* array, std::size_t ld) -> void {
     log_matrix(level, s, m, n, reinterpret_cast<const std::complex<double>*>(array), ld);
   }
#endif

 private:
   static auto convert_level(BippLogLevel l) -> spdlog::level::level_enum;

   BippLogLevel level_;
   std::shared_ptr<spdlog::logger> logger_;
   rt_graph::Timer timer_;
};
}  // namespace bipp
