#include "bridge.h"
#include <cmath>
using namespace sputniktorch;
extern "C" {
// The algorithm is an Sputnik callback. Keep its short LibTorch calls on one
// blocking worker instead of parking the caller for every tensor operation.
SputnikStatus amt_execution_batch(SputnikCtx *cx, const SputnikValue *a, size_t n,
                                SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    if (sputnik_call_block(cx, a[0], nullptr, 0, out) != SPUTNIK_OK)
      throw FaultPending{};
  });
}
SputnikStatus amt_optimizer_validate(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 2);
    require(sputnik_is_list(cx, a[0]), "TypeError", "expected parameter handles");
    const auto lr = number(cx, a[1]);
    require(std::isfinite(lr) && lr >= 0, "ArgumentError",
            "learning rate must be finite and nonnegative");
    for (size_t i = 0; i < sputnik_list_len(cx, a[0]); ++i) {
      auto &c = core(cx, sputnik_list_at(cx, a[0], i));
      writable(c);
      auto &p = c.tensor;
      require(p.is_leaf() && p.is_floating_point(), "Sputniktorch.Errors.AutogradError",
              "optimizer requires floating leaf parameters");
      if (p.grad().defined()) {
        require(p.sizes() == p.grad().sizes() && p.device() == p.grad().device() &&
                    p.scalar_type() == p.grad().scalar_type() && p.grad().layout() == at::kStrided,
                "Sputniktorch.Errors.AutogradError", "optimizer requires matching dense gradients");
      }
    }
    *out = sputnik_make_null(cx);
  });
}
SputnikStatus amt_optimizer_coefficients(SputnikCtx *cx, const SputnikValue *a, size_t n,
                                       SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    require(sputnik_is_list(cx, a[0]), "TypeError", "expected numeric coefficients");
    for (size_t i = 0; i < sputnik_list_len(cx, a[0]); ++i) {
      require(std::isfinite(number(cx, sputnik_list_at(cx, a[0], i))), "ArgumentError",
              "optimizer coefficients must be finite");
    }
    *out = sputnik_make_null(cx);
  });
}
}
