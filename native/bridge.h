#pragma once
#include "runtime/sputnik_ext.h"
#include "onnx/tracing.h"
#include "boundary_profile.h"
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace sputniktorch {
inline constexpr char tensor_tag[] = "sputniktorch.TensorCore";
struct Error : std::runtime_error {
  const char *kind;
  Error(const char *kind, const std::string &message) : std::runtime_error(message), kind(kind) {}
};
struct FaultPending {};
struct TensorCore {
  at::Tensor tensor;
  bool readonly = false;
};
struct ExecutionGuard {
  c10::InferenceMode inference;
  at::AutoGradMode grad;
  TraceKernelGuard trace;
  explicit ExecutionGuard(int64_t mode, bool mutating = false)
      : inference((mode & 3) == 2), grad((mode & 3) == 0), trace(mode / 4) {
    if (mode < 0 || (mode & 3) == 3)
      throw Error("ArgumentError", "invalid execution mode");
    if (mutating && mode >= 4)
      throw Error("Sputniktorch.Errors.ONNXError", "tensor mutation is forbidden during ONNX export");
  }
};
inline void require(bool cond, const char *kind, const std::string &message) {
  if (!cond)
    throw Error(kind, message);
}
inline void require(bool cond, const char *kind, const char *message) {
  // Literal diagnostics must not allocate a std::string on successful calls.
  if (!cond)
    throw Error(kind, message);
}
inline std::string string(SputnikCtx *cx, SputnikValue value) {
  const char *p;
  size_t n;
  require(sputnik_str_view(cx, value, &p, &n), "TypeError", "expected Str");
  return std::string(p, n);
}
inline int64_t integer(SputnikCtx *cx, SputnikValue value) {
  int64_t v;
  require(sputnik_as_int(cx, value, &v), "TypeError", "expected Int");
  return v;
}
inline double number(SputnikCtx *cx, SputnikValue value) {
  int64_t i;
  double f;
  if (sputnik_as_int(cx, value, &i))
    return static_cast<double>(i);
  require(sputnik_as_float(cx, value, &f), "TypeError", "expected numeric scalar");
  return f;
}
inline bool boolean(SputnikCtx *cx, SputnikValue value) {
  int v;
  require(sputnik_as_bool(cx, value, &v), "TypeError", "expected Bool");
  return v != 0;
}
inline std::vector<int64_t> integers(SputnikCtx *cx, SputnikValue value) {
  require(sputnik_is_list(cx, value), "TypeError", "expected List[Int]");
  const size_t count = sputnik_list_len(cx, value);
  std::vector<int64_t> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i)
    out.push_back(integer(cx, sputnik_list_at(cx, value, i)));
  return out;
}
inline TensorCore &core(SputnikCtx *cx, SputnikValue value) {
  void *ptr = nullptr;
  if (!sputnik_handle_ptr(cx, value, tensor_tag, &ptr))
    throw FaultPending{};
  return *static_cast<TensorCore *>(ptr);
}
inline SputnikValue wrap(SputnikCtx *cx, at::Tensor tensor, bool readonly = false) {
  auto ptr = std::make_unique<TensorCore>(TensorCore{std::move(tensor), readonly});
  SputnikValue value = sputnik_make_handle(cx, tensor_tag, ptr.get());
  if (!value)
    throw FaultPending{};
  ptr.release();
  return value;
}
inline void writable(const TensorCore &value) {
  require(!value.readonly, "Sputniktorch.Errors.ReadOnlyTensorError",
          "snapshot and its aliases are read-only; use copy()");
}
inline SputnikValue text(SputnikCtx *cx, const std::string &value) {
  return sputnik_make_str(cx, value.data(), value.size());
}
inline void argc_check(size_t actual, size_t expected) {
  require(actual == expected, "ArgumentError", "wrong number of native arguments");
}
template <class F> SputnikStatus boundary(
    SputnikCtx *cx, F &&f, const char *caller = __builtin_FUNCTION()) noexcept {
  BoundaryProfileScope profile(caller);
  try {
    f();
    return SPUTNIK_OK;
  } catch (const FaultPending &) {
    return SPUTNIK_ERR;
  } catch (const Error &e) {
    return sputnik_fault(cx, e.kind, e.what());
  } catch (const c10::Error &e) {
    return sputnik_fault(cx, "Sputniktorch.Errors.Error", e.what_without_backtrace());
  } catch (const std::bad_alloc &) {
    return sputnik_fault(cx, "Sputniktorch.Errors.Error", "native allocation failed");
  } catch (const std::exception &e) {
    return sputnik_fault(cx, "Sputniktorch.Errors.Error", e.what());
  } catch (...) {
    return sputnik_fault(cx, "Sputniktorch.Errors.Error", "unknown native exception");
  }
}
} // namespace sputniktorch
