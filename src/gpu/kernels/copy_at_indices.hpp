#pragma once

#include <cstddef>

#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {

// copy matrix elements at any index combination in "indices" to output. For square matrices only.
template <typename T>
auto copy_matrix_from_indices(const api::DevicePropType& prop, const api::StreamType& stream,
                                     std::size_t n, const std::size_t* indices, const T* a,
                                     std::size_t lda, T* b, std::size_t ldb) -> void;

// copy rows to row indices in "rowIndices" of output
template <typename T>
auto copy_matrix_rows_to_indices(const api::DevicePropType& prop, const api::StreamType& stream,
                                 std::size_t nRows, std::size_t nCols,
                                 const std::size_t* rowIndices, const T* a, std::size_t lda, T* b,
                                 std::size_t ldb) -> void;
}
}  // namespace bipp
