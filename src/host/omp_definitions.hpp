#pragma once

#include "bipp/config.h"

#ifdef BIPP_OMP
#include <omp.h>
#define BIPP_OMP_PRAGMA(content) _Pragma(content)

#else
#define BIPP_OMP_PRAGMA(content)
namespace bipp {
inline int omp_get_num_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline int omp_get_max_threads() { return 1; }
inline int omp_in_parallel() { return 0; }
inline int omp_get_nested() { return 0; }
inline int omp_get_num_procs() { return 1; }
inline int omp_get_level() { return 0; }
inline void omp_set_nested(int) {}
}  // namespace bipp
#endif
