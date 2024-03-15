#pragma once

#include "bipp/config.h"
#include "context_internal.hpp"
#include "gpu/util/runtime_api.hpp"

namespace bipp {
namespace gpu {
namespace eigensolver {

auto solve(Queue& queue, char jobz, char uplo, int n, api::ComplexType<float>* a, int lda,
           float* w) -> void;

auto solve(Queue& queue, char jobz, char uplo, int n, api::ComplexType<double>* a, int lda,
           double* w) -> void;

auto solve(Queue& queue, char jobz, char uplo, int n, api::ComplexType<float>* a, int lda,
           api::ComplexType<float>* b, int ldb, float* w) -> void;

auto solve(Queue& queue, char jobz, char uplo, int n, api::ComplexType<double>* a, int lda,
           api::ComplexType<double>* b, int ldb, double* w) -> void;


}  // namespace eigensolver

}  // namespace gpu
}  // namespace bipp
