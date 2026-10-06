#include "raw_calls.h"

int64_t bench_raw_constant() { return 42; }
int64_t bench_raw_identity(int64_t value) { return value; }
int64_t bench_raw_numel(const at::Tensor &tensor) { return tensor.numel(); }
at::Tensor bench_raw_relu(const at::Tensor &tensor) { return at::relu(tensor); }
