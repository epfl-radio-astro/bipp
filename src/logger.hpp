#pragma once

#include <any>
#include <complex>
#include <cstddef>
#include <memory>
#include <string>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "memory/view.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/runtime_api.hpp"
#endif

#include <spdlog/logger.h>

namespace bipp {

class Logger {
public:
   explicit Logger(BippLogLevel level, const char* out = "stdout");

   Logger(const Logger&) = delete;

   Logger(Logger&&) = default;

   auto operator=(const Logger&) -> Logger& = delete;

   auto operator=(Logger&&) -> Logger& = default;

   template <typename... Args>
   void log(BippLogLevel level, const std::string_view& s, Args&&... args) {
     if (level <= level_) logger_->log(convert_level(level), s, std::forward<Args>(args)...);
   }

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
   inline auto log_matrix(BippLogLevel level, const std::string_view& s,
                          ConstHostView<T, 2> array) {
     this->log_matrix(level, s, array.shape()[0], array.shape()[1], array.data(),
                      array.strides()[1]);
   }

   template <typename T>
   inline auto log_matrix(BippLogLevel level, const std::string_view& s,
                          ConstHostView<T, 1> array) {
     this->log_matrix(level, s, array.shape()[0], 1, array.data(), array.shape()[0]);
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
};
}  // namespace bipp
