#include "bridge.h"

using namespace sputniktorch;

// Sputnik owns composition, registries and mode policy; this boundary only
// marshals tensors/configuration into stable ATen kernels.
extern "C" SputnikStatus amt_nn(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]));
    const auto op = string(cx, a[0]);
    require(sputnik_is_list(cx, a[1]) && sputnik_is_list(cx, a[2]), "TypeError",
            "expected tensor and configuration lists");
    auto input = [&](size_t i) {
      require(i < sputnik_list_len(cx, a[1]), "ArgumentError", "missing kernel input");
      auto v = sputnik_list_at(cx, a[1], i);
      return sputnik_is_null(cx, v) ? at::Tensor() : core(cx, v).tensor;
    };
    auto arg = [&](size_t i) {
      require(i < sputnik_list_len(cx, a[2]), "ArgumentError", "missing kernel option");
      return sputnik_list_at(cx, a[2], i);
    };
    auto ints = [&](size_t i) { return integers(cx, arg(i)); };
    auto num = [&](size_t i) { return number(cx, arg(i)); };
    auto integer_arg = [&](size_t i) { return integer(cx, arg(i)); };
    auto bool_arg = [&](size_t i) { return boolean(cx, arg(i)); };
    const auto x = input(0);
    at::Tensor r;
    if (op == "gelu")
      r = at::gelu(x, string(cx, arg(0)));
    else if (op == "silu")
      r = at::silu(x);
    else if (op == "mish")
      r = at::mish(x);
    else if (op == "leaky_relu")
      r = at::leaky_relu(x, num(0));
    else if (op == "elu")
      r = at::elu(x, num(0));
    else if (op == "selu")
      r = at::selu(x);
    else if (op == "softplus")
      r = at::softplus(x, num(0), num(1));
    else if (op == "hardtanh")
      r = at::hardtanh(x, num(0), num(1));
    else if (op == "hardsigmoid")
      r = at::hardsigmoid(x);
    else if (op == "hardswish")
      r = at::hardswish(x);
    else if (op == "log_sigmoid")
      r = at::log_sigmoid(x);
    else if (op == "softshrink")
      r = at::softshrink(x, num(0));
    else if (op == "hardshrink")
      r = at::hardshrink(x, num(0));
    else if (op == "threshold")
      r = at::threshold(x, num(0), num(1));
    else if (op == "prelu")
      r = at::prelu(x, input(1));
    else if (op == "bilinear")
      r = at::bilinear(x, input(1), input(2), input(3));
    else if (op == "pixel_shuffle")
      r = at::pixel_shuffle(x, integer_arg(0));
    else if (op == "pixel_unshuffle")
      r = at::pixel_unshuffle(x, integer_arg(0));
    else if (op == "unfold")
      r = at::im2col(x, ints(0), ints(1), ints(2), ints(3));
    else if (op == "fold")
      r = at::col2im(x, ints(0), ints(1), ints(2), ints(3), ints(4));
    else if (op == "embedding")
      r = at::embedding(input(1), x, integer_arg(0), false, false);
    else if (op == "conv1d")
      r = at::conv1d(x, input(1), input(2), ints(0), ints(1), ints(2), integer_arg(3));
    else if (op == "conv2d")
      r = at::conv2d(x, input(1), input(2), ints(0), ints(1), ints(2), integer_arg(3));
    else if (op == "conv3d")
      r = at::conv3d(x, input(1), input(2), ints(0), ints(1), ints(2), integer_arg(3));
    else if (op == "conv_transpose1d")
      r = at::conv_transpose1d(x, input(1), input(2), ints(0), ints(1), ints(2), integer_arg(3),
                               ints(4));
    else if (op == "conv_transpose2d")
      r = at::conv_transpose2d(x, input(1), input(2), ints(0), ints(1), ints(2), integer_arg(3),
                               ints(4));
    else if (op == "conv_transpose3d")
      r = at::conv_transpose3d(x, input(1), input(2), ints(0), ints(1), ints(2), integer_arg(3),
                               ints(4));
    else if (op == "max_pool1d")
      r = at::max_pool1d(x, ints(0), ints(1), ints(2), ints(3), bool_arg(4));
    else if (op == "max_pool2d")
      r = at::max_pool2d(x, ints(0), ints(1), ints(2), ints(3), bool_arg(4));
    else if (op == "max_pool3d")
      r = at::max_pool3d(x, ints(0), ints(1), ints(2), ints(3), bool_arg(4));
    else if (op == "avg_pool1d")
      r = at::avg_pool1d(x, ints(0), ints(1), ints(2), bool_arg(3), bool_arg(4));
    else if (op == "avg_pool2d")
      r = at::avg_pool2d(x, ints(0), ints(1), ints(2), bool_arg(3), bool_arg(4));
    else if (op == "avg_pool3d")
      r = at::avg_pool3d(x, ints(0), ints(1), ints(2), bool_arg(3), bool_arg(4));
    else if (op == "adaptive_avg_pool1d")
      r = at::adaptive_avg_pool1d(x, ints(0));
    else if (op == "adaptive_avg_pool2d")
      r = at::adaptive_avg_pool2d(x, ints(0));
    else if (op == "adaptive_avg_pool3d")
      r = at::adaptive_avg_pool3d(x, ints(0));
    else if (op == "adaptive_max_pool1d")
      r = std::get<0>(at::adaptive_max_pool1d(x, ints(0)));
    else if (op == "adaptive_max_pool2d")
      r = std::get<0>(at::adaptive_max_pool2d(x, ints(0)));
    else if (op == "adaptive_max_pool3d")
      r = std::get<0>(at::adaptive_max_pool3d(x, ints(0)));
    else if (op == "layer_norm")
      r = at::layer_norm(x, ints(0), input(1), input(2), num(1));
    else if (op == "group_norm")
      r = at::group_norm(x, integer_arg(0), input(1), input(2), num(1));
    else if (op == "batch_norm") {
      const bool training = bool_arg(0);
      require(!training || (!input(3).defined() && !input(4).defined()), "ArgumentError",
              "training with running buffers requires batch_norm!");
      r = at::batch_norm(x, input(1), input(2), input(3), input(4), training, num(1), num(2),
                         false);
    } else if (op == "pad")
      r = at::pad(x, ints(0), string(cx, arg(1)), num(2));
    else if (op == "cat" || op == "stack") {
      std::vector<at::Tensor> tensors;
      for (size_t i = 0; i < sputnik_list_len(cx, a[1]); ++i)
        tensors.push_back(input(i));
      r = op == "cat" ? at::cat(tensors, integer_arg(0)) : at::stack(tensors, integer_arg(0));
    } else if (op == "where")
      r = at::where(x, input(1), input(2));
    else if (op == "masked_softmax") {
      const auto mask = input(1);
      require(mask.scalar_type() == at::kBool, "Sputniktorch.Errors.DTypeError",
              "attention mask must be boolean");
      const auto dim = integer_arg(0);
      auto scores = x.masked_fill(mask.logical_not(), -std::numeric_limits<double>::infinity());
      scores = at::where(mask.any(dim, true), scores, at::zeros_like(scores));
      r = scores.softmax(dim).masked_fill(mask.logical_not(), 0);
    } else if (op == "cross_entropy")
      r = at::cross_entropy_loss(x, input(1), input(2), integer_arg(0), integer_arg(1), num(2));
    else if (op == "nll_loss")
      r = at::nll_loss_nd(x, input(1), input(2), integer_arg(0), integer_arg(1));
    else if (op == "bce_logits")
      r = at::binary_cross_entropy_with_logits(x, input(1), input(2), input(3), integer_arg(0));
    else if (op == "bce")
      r = at::binary_cross_entropy(x, input(1), input(2), integer_arg(0));
    else if (op == "smooth_l1")
      r = at::smooth_l1_loss(x, input(1), integer_arg(0), num(1));
    else if (op == "huber")
      r = at::huber_loss(x, input(1), integer_arg(0), num(1));
    else if (op == "kl_div")
      r = at::kl_div(x, input(1), integer_arg(0), bool_arg(1));
    else
      throw Error("ArgumentError", "unsupported NN kernel: " + op);
    *out = wrap(cx, std::move(r));
  });
}

extern "C" SputnikStatus amt_batch_norm(SputnikCtx *cx, const SputnikValue *a, size_t n,
                                      SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 9);
    ExecutionGuard mode(integer(cx, a[8]), boolean(cx, a[5]));
    auto tensor = [&](size_t i) {
      return sputnik_is_null(cx, a[i]) ? at::Tensor() : core(cx, a[i]).tensor;
    };
    for (size_t i : {3U, 4U})
      if (!sputnik_is_null(cx, a[i]))
        writable(core(cx, a[i]));
    *out = wrap(cx, at::batch_norm(tensor(0), tensor(1), tensor(2), tensor(3), tensor(4),
                                   boolean(cx, a[5]), number(cx, a[6]), number(cx, a[7]), false));
  });
}
