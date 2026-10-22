#include "bridge.h"

// Only ATen primitives live here; layout, mask composition and RNG are Sputnik.
extern "C" SputnikStatus amt_attention(SputnikCtx *cx, const SputnikValue *a,
                                       size_t n, SputnikValue *out) {
  return sputnik_torch::boundary(cx, [&] {
    using namespace sputnik_torch;
    argc_check(n, 4);
    const auto mode = integer(cx, a[3]);
    ExecutionGuard guard(mode);
    const auto op = string(cx, a[0]);
    auto input = [&](size_t i) {
      require(i < sputnik_list_len(cx, a[1]), "ArgumentError", "missing attention input");
      auto value = sputnik_list_at(cx, a[1], i);
      return sputnik_is_null(cx, value) ? at::Tensor() : core(cx, value).tensor;
    };
    const auto x = input(0);
    at::Tensor result;
    if (op == "sdpa") {
      const auto mask = input(3);
      const auto scale = sputnik_list_at(cx, a[2], 0);
      result = at::scaled_dot_product_attention(x, input(1), input(2),
        mask.defined() ? std::make_optional(mask) : std::nullopt, 0, false,
        sputnik_is_null(cx, scale) ? std::nullopt : std::make_optional(number(cx, scale)));
    } else if (op == "mask_bias") {
      result = at::zeros(x.sizes(), input(1).options())
        .masked_fill(x.logical_not(), -std::numeric_limits<double>::infinity());
    } else if (op == "softmax") {
      // Avoid NaN softmax intermediates, including their backward, for empty rows.
      auto valid = x.ne(-std::numeric_limits<double>::infinity()).any(-1, true);
      auto scores = at::where(valid, x, at::zeros_like(x));
      result = scores.softmax(-1).masked_fill(valid.logical_not(), 0);
    } else {
      throw Error("ArgumentError", "unknown attention primitive");
    }
    *out = wrap(cx, result);
  });
}
