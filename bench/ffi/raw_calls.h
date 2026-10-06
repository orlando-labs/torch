#pragma once
#include <ATen/ATen.h>
#include <cstdint>

// Separate translation unit, no LTO: the benchmark loop must actually call
// these functions, including constant/identity controls.
int64_t bench_raw_constant();
int64_t bench_raw_identity(int64_t value);
int64_t bench_raw_numel(const at::Tensor &tensor);
at::Tensor bench_raw_relu(const at::Tensor &tensor);
