#pragma once

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
namespace eigensolver {
auto solve(ContextInternal& ctx, char jobz, char range, char uplo, int n,
           api::ComplexType<float>* a, int lda, float vl, float vu, int il, int iu, int* m,
           float* w) -> void;

auto solve(ContextInternal& ctx, char jobz, char range, char uplo, int n,
           api::ComplexType<double>* a, int lda, double vl, double vu, int il, int iu, int* m,
           double* w) -> void;

auto solve(ContextInternal& ctx, char jobz, char range, char uplo, int n,
           api::ComplexType<float>* a, int lda, api::ComplexType<float>* b, int ldb, float vl,
           float vu, int il, int iu, int* m, float* w) -> void;

auto solve(ContextInternal& ctx, char jobz, char range, char uplo, int n,
           api::ComplexType<double>* a, int lda, api::ComplexType<double>* b, int ldb, double vl,
           double vu, int il, int iu, int* m, double* w) -> void;
}  // namespace eigensolver

}  // namespace gpu
}  // namespace bipp
