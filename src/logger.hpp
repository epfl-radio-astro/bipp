#pragma once

#include <any>
#include <complex>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <optional>

#include "bipp/config.h"
#include "bipp/enums.h"
#include "memory/view.hpp"
#include "rt_graph.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/runtime_api.hpp"
#endif

#include <spdlog/logger.h>

namespace bipp {

class Logger {
public:
  Logger();

  Logger(const Logger&) = delete;

  Logger(Logger&&) = default;

  auto operator=(const Logger&) -> Logger& = delete;

  auto operator=(Logger&&) -> Logger& = default;

  template <typename... Args>
  auto log(BippLogLevel level, const std::string_view& s, Args&&... args) -> void {
    if (level <= level_ && data_.has_value())
      data_.value().logger->log(convert_level(level), s, std::forward<Args>(args)...);
  }

  // Internally store timings. Does not print to log until log_timings() is called
  template <std::size_t N>
  auto scoped_timing(BippLogLevel level, const char (&tag)[N]) -> rt_graph::ScopedTiming {
    if (level <= level_ && data_.has_value()) {
      return rt_graph::ScopedTiming(tag, data_.value().timer);
    }
    return rt_graph::ScopedTiming();
  }

  auto scoped_timing(BippLogLevel level, std::string tag) -> rt_graph::ScopedTiming {
    if (level <= level_ && data_.has_value()) {
      return rt_graph::ScopedTiming(std::move(tag), data_.value().timer);
    }
    return rt_graph::ScopedTiming();
  }

  template <std::size_t N>
  auto start_timing(BippLogLevel level, const char (&tag)[N]) -> void {
    if (level <= level_ && data_.has_value()) {
      data_.value().timer.start(tag);
    }
  }

  auto start_timing(BippLogLevel level, std::string tag) -> void {
    if (level <= level_ && data_.has_value()) {
      data_.value().timer.start(tag);
    }
  }

  template <std::size_t N>
  auto stop_timing(BippLogLevel level, const char (&tag)[N]) -> void {
    if (level <= level_ && data_.has_value()) {
      data_.value().timer.stop(tag);
    }
  }

  auto stop_timing(BippLogLevel level, std::string tag) -> void {
    if (level <= level_ && data_.has_value()) {
      data_.value().timer.stop(tag);
    }
  }

  auto log_timings() -> void;

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
    this->log_matrix(level, s, array.shape(0), array.shape(1), array.data(), array.strides(1));
  }

  template <typename T>
  inline auto log_matrix(BippLogLevel level, const std::string_view& s, ConstView<T, 1> array) {
    this->log_matrix(level, s, array.shape(), 1, array.data(), array.shape());
  }

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  inline auto log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m,
                         std::size_t n, const gpu::api::ComplexType<float>* array, std::size_t ld)
      -> void {
    log_matrix(level, s, m, n, reinterpret_cast<const std::complex<float>*>(array), ld);
  }
  inline auto log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m,
                         std::size_t n, const gpu::api::ComplexType<double>* array, std::size_t ld)
      -> void {
    log_matrix(level, s, m, n, reinterpret_cast<const std::complex<double>*>(array), ld);
  }
#endif

  ~Logger() {
   try {
     log_timings();
   } catch (...) {
   }
  }

private:
  static auto convert_level(BippLogLevel l) -> spdlog::level::level_enum;

  struct LogObjects {
    std::shared_ptr<spdlog::logger> logger;
    rt_graph::Timer timer;
  };

  BippLogLevel level_ = BIPP_LOG_LEVEL_OFF;
  // Avoid any allocation if turned off due to usage in global variable.
  std::optional<LogObjects> data_;
};

// declare global logger
extern Logger globLogger;
}  // namespace bipp
