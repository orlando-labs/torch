#include "bridge.h"
#include "raw_calls.h"

using namespace sputnik_torch;

extern "C" {
// Minimal ABI controls deliberately omit the production exception/validation
// wrapper. Compare them with the checked calls below; never use them in APIs.
SputnikStatus amt_bench_constant(SputnikCtx *cx, const SputnikValue *, size_t,
                               SputnikValue *out) {
  *out = sputnik_make_int(cx, 42);
  return SPUTNIK_OK;
}
SputnikStatus amt_bench_identity(SputnikCtx *, const SputnikValue *args, size_t,
                               SputnikValue *out) {
  *out = args[0];
  return SPUTNIK_OK;
}
SputnikStatus amt_bench_int(SputnikCtx *cx, const SputnikValue *args, size_t n,
                          SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    *out = sputnik_make_int(cx, integer(cx, args[0]));
  });
}
SputnikStatus amt_bench_handle(SputnikCtx *cx, const SputnikValue *args, size_t n,
                             SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    core(cx, args[0]);
    *out = args[0];
  });
}
SputnikStatus amt_bench_numel(SputnikCtx *cx, const SputnikValue *args, size_t n,
                            SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    *out = sputnik_make_int(cx, core(cx, args[0]).tensor.numel());
  });
}
SputnikStatus amt_bench_string(SputnikCtx *cx, const SputnikValue *args, size_t n,
                             SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    *out = sputnik_make_int(cx, string(cx, args[0]).size());
  });
}
SputnikStatus amt_bench_list_len(SputnikCtx *cx, const SputnikValue *args, size_t n,
                               SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    require(sputnik_is_list(cx, args[0]), "TypeError", "expected List");
    *out = sputnik_make_int(cx, sputnik_list_len(cx, args[0]));
  });
}
SputnikStatus amt_bench_list_sum(SputnikCtx *cx, const SputnikValue *args, size_t n,
                               SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    // This is the actual adapter's List[Int] conversion, with its temporary
    // vector, per-element readers and arena entries. Use small, safe values.
    int64_t sum = 0;
    for (int64_t value : integers(cx, args[0])) sum += value;
    *out = sputnik_make_int(cx, sum);
  });
}
SputnikStatus amt_bench_relu(SputnikCtx *cx, const SputnikValue *args, size_t n,
                           SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    ExecutionGuard guard(0);
    *out = wrap(cx, at::relu(core(cx, args[0]).tensor));
  });
}
SputnikStatus amt_bench_cpp_loop(SputnikCtx *cx, const SputnikValue *args, size_t n,
                               SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 3);
    const std::string op = string(cx, args[0]);
    const auto &tensor = core(cx, args[1]).tensor;
    const int64_t count = integer(cx, args[2]);
    require(count > 0, "ArgumentError", "positive iteration count required");
    ExecutionGuard guard(0);
    int64_t value = 0;
    if (op == "constant") {
      for (int64_t i = 0; i < count; ++i) value = bench_raw_constant();
    } else if (op == "identity") {
      for (int64_t i = 0; i < count; ++i) value = bench_raw_identity(42);
    } else if (op == "numel") {
      for (int64_t i = 0; i < count; ++i) value = bench_raw_numel(tensor);
    } else if (op == "relu") {
      at::Tensor result;
      for (int64_t i = 0; i < count; ++i) result = bench_raw_relu(tensor);
      *out = wrap(cx, std::move(result));
      return;
    } else {
      throw Error("ArgumentError", "unknown benchmark operation");
    }
    *out = sputnik_make_int(cx, value);
  });
}
} // extern "C"
