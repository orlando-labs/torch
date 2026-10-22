#pragma once
#include <cstdint>
#include <memory>

namespace sputnik_torch {
// A snapshot of the logical Sputnik task reattaches tracing on each FFI worker.
// No tracer state is left on a carrier thread after the kernel returns.
class TraceKernelGuard {
  struct Impl;
  std::unique_ptr<Impl> impl_;
public:
  explicit TraceKernelGuard(int64_t token);
  ~TraceKernelGuard();
};
}
