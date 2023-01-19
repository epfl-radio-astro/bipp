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

#ifndef __cplusplus
/*! \cond PRIVATE */
// C only
typedef enum BippProcessingUnit BippProcessingUnit;
typedef enum BippFilter BippFilter;
/*! \endcond */
#endif  // cpp
