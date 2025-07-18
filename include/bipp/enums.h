#pragma once

#include <bipp/config.h>

enum BippProcessingUnit { BIPP_PU_AUTO, BIPP_PU_CPU, BIPP_PU_GPU };

enum BippPrecision { BIPP_PRECISION_SINGLE, BIPP_PRECISION_DOUBLE };

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
typedef enum BippLogLevel BippLogLevel;
typedef enum BippPrecision BippPrecision;
/*! \endcond */
#endif  // cpp
