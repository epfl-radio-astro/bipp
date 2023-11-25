#include "logger.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/device_pointer.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

namespace bipp {
namespace {

template<typename T>
struct ArrayInfo {
  ArrayInfo(std::size_t m, std::size_t n, const T* array, std::size_t ld)
      : m(m), n(n), sum(0), normal(false), subnormal(false), inf(false), nan(false), zero(false) {
    std::vector<T> buffer;
    const T* data = array;

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    if (gpu::is_device_ptr(array)) {
      buffer.resize(m * n);
      gpu::api::device_synchronize();
      gpu::api::memcpy_2d(buffer.data(), m * sizeof(T), array, ld * sizeof(T), m * sizeof(T), n,
                          gpu::api::flag::MemcpyDefault);

      data = buffer.data();
      ld = m;
    }
#endif

    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
      min = T{std::numeric_limits<typename T::value_type>::max(),
              std::numeric_limits<typename T::value_type>::max()};
      max = T{-std::numeric_limits<typename T::value_type>::max(),
              -std::numeric_limits<typename T::value_type>::max()};

      for(std::size_t c = 0; c < n; ++c) {
        for (std::size_t r = 0; r < m; ++r) {
          const auto& val = data[c * ld + r];
          min = T{std::min(min.real(), val.real()), std::min(min.imag(), val.imag())};
          max = T{std::max(max.real(), val.real()), std::max(max.imag(), val.imag())};

          sum += val;

          const auto fpReal = std::fpclassify(val.real());
          const auto fpImag = std::fpclassify(val.imag());

          normal |= (fpReal == FP_NORMAL) | (fpImag == FP_NORMAL);
          subnormal |= (fpReal == FP_SUBNORMAL) | (fpImag == FP_SUBNORMAL);
          inf |= (fpReal == FP_INFINITE) | (fpImag == FP_INFINITE);
          nan |= (fpReal == FP_NAN) | (fpImag == FP_NAN);
          zero |= (fpReal == FP_ZERO) | (fpImag == FP_ZERO);
        }
      }
    } else {
      min = std::numeric_limits<T>::max();
      max = -std::numeric_limits<T>::max();

      for(std::size_t c = 0; c < n; ++c) {
        for (std::size_t r = 0; r < m; ++r) {
          const auto& val = data[c * ld + r];
          min = std::min(min, val);
          max = std::max(max, val);

          sum += val;

          const auto fp = std::fpclassify(val);

          normal |= (fp == FP_NORMAL);
          subnormal |= (fp == FP_SUBNORMAL);
          inf |= (fp == FP_INFINITE);
          nan |= (fp == FP_NAN);
        }
      }

    }
  }

  std::size_t m, n;
  T min, max, sum;
  bool normal, subnormal, inf, nan, zero;
};

template <typename T>
auto log_array(spdlog::level::level_enum lvl, spdlog::logger& log, const std::string_view& s,
               std::size_t m, std::size_t n, const T* array, std::size_t ld) -> void {

  std::string logString;

  logString += "array \"" + std::string(s) + "\": ";

  if(m == 0 || n == 0) {
    logString += " size ({}, {})";
    log.log(lvl, logString, m, n);
    return;
  }

  if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
    logString += " size ({}, {}), min ({}, {}), max ({}, {}), sum ({}, {})";
  } else {
    logString += " size ({}, {}), min {}, max {}, sum {}";
  }

  ArrayInfo<T> info(m, n, array, ld);

  logString += ", fp classes [";

  if(info.normal) logString += "normal,";
  if (info.zero) logString += "zero,";
  if (info.subnormal) logString += "subnormal,";
  if (info.inf) logString += "inf,";
  if (info.nan) logString += "nan,";

  if(logString.back() == ',') logString.pop_back();
  logString += "]";

  if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
    log.log(lvl, logString, info.m, info.n, info.min.real(), info.min.imag(),
            info.max.real(), info.max.imag(), info.sum.real(), info.sum.imag());
  } else {
    log.log(lvl,  logString, info.m, info.n, info.min, info.max, info.sum);
  }
}

}  // namespace


Logger::Logger(BippLogLevel level, const char* out) : level_(level) {
  spdlog::set_automatic_registration(false);
  if (!std::strcmp(out, "stdout")) {
      logger_ = spdlog::stdout_logger_st("bipp");
  } else if (!std::strcmp(out, "stderr")) {
      logger_ = spdlog::stderr_logger_st("bipp");
  } else {
      try {
      logger_ = spdlog::basic_logger_st("bipp", out, false);
      } catch (const spdlog::spdlog_ex& ex) {
      logger_ = spdlog::stderr_logger_st("bipp");
      }
  }
  logger_->set_level(convert_level(level));
}

auto Logger::convert_level(BippLogLevel l) -> spdlog::level::level_enum {
  switch (l) {
    case BippLogLevel::BIPP_LOG_LEVEL_OFF:
      return spdlog::level::off;
    case BippLogLevel::BIPP_LOG_LEVEL_ERROR:
      return spdlog::level::err;
    case BippLogLevel::BIPP_LOG_LEVEL_WARN:
      return spdlog::level::warn;
    case BippLogLevel::BIPP_LOG_LEVEL_INFO:
      return spdlog::level::info;
    case BippLogLevel::BIPP_LOG_LEVEL_DEBUG:
      return spdlog::level::debug;
  }

  return spdlog::level::debug;
}

auto Logger::log_timings(BippLogLevel level) -> void {
  if (level > level_ || timer_.empty()) return;
    const auto result = timer_.process();
    auto msg = result.print();
    log(level, "\n {} \n", msg);
}

auto Logger::log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m, std::size_t n,
                   const float* array, std::size_t ld) -> void {
  if (level <= level_) {
      log_array(convert_level(level), *logger_, s, m, n, array, ld);
  }
}
auto Logger::log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m, std::size_t n,
                   const double* array, std::size_t ld) -> void {
  if (level <= level_) {
      log_array(convert_level(level), *logger_, s, m, n, array, ld);
  }
}
auto Logger::log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m, std::size_t n,
                   const std::complex<float>* array, std::size_t ld) -> void {
  if (level <= level_) {
      log_array(convert_level(level), *logger_, s, m, n, array, ld);
  }
}
auto Logger::log_matrix(BippLogLevel level, const std::string_view& s, std::size_t m, std::size_t n,
                        const std::complex<double>* array, std::size_t ld) -> void {
  if (level <= level_) {
      log_array(convert_level(level), *logger_, s, m, n, array, ld);
  }
}

}  // namespace bipp
