#pragma once

#include <bipp/config.h>

enum BippProcessingUnit { BIPP_PU_AUTO, BIPP_PU_CPU, BIPP_PU_GPU };

enum BippFilter {
  BIPP_FILTER_LSQ,
  BIPP_FILTER_STD,
  BIPP_FILTER_SQRT,
  BIPP_FILTER_INV,
  BIPP_FILTER_INV_SQ
};

enum BippLogLevel {
  BIPP_LOG_LEVEL_OFF,
  BIPP_LOG_LEVEL_ERROR,
  BIPP_LOG_LEVEL_WARN,
  BIPP_LOG_LEVEL_INFO,
  BIPP_LOG_LEVEL_DEBUG,
};

#ifndef __cplusplus
/*! \cond PRIVATE */
// C only
typedef enum BippProcessingUnit BippProcessingUnit;
typedef enum BippFilter BippFilter;
typedef enum BippLogLevel BippLogLevel;
/*! \endcond */
#endif  // cpp
