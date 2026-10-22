#include "bridge.h"
#include <torch/nn/functional/activation.h>

// CPU oracle linked only into these tests, never into the package adapter.
extern "C" SputnikStatus amt_attention_reference(SputnikCtx *cx, const SputnikValue *a,
                                                 size_t n, SputnikValue *out) {
  return sputnik_torch::boundary(cx, [&] {
    using namespace sputnik_torch;
    argc_check(n, 5);
    ExecutionGuard mode(0);
    auto leaf = [&](SputnikValue value) {
      if (sputnik_is_null(cx, value)) return at::Tensor();
      auto result = core(cx, value).tensor.detach().to(at::kCPU).clone();
      if (result.is_floating_point()) result.set_requires_grad(true);
      return result;
    };
    std::vector<at::Tensor> inputs, parameters;
    for (size_t i = 0; i < sputnik_list_len(cx, a[1]); ++i)
      inputs.push_back(leaf(sputnik_list_at(cx, a[1], i)));
    for (size_t i = 0; i < sputnik_list_len(cx, a[2]); ++i)
      parameters.push_back(leaf(sputnik_list_at(cx, a[2], i)));
    auto mask = leaf(a[3]);
    auto option = [&](size_t i) { return sputnik_list_at(cx, a[4], i); };
    at::Tensor result, weights;
    if (string(cx, a[0]) == "sdpa") {
      auto scale = option(0);
      result = at::scaled_dot_product_attention(inputs[0], inputs[1], inputs[2],
        mask.defined() ? std::make_optional(mask) : std::nullopt, 0, boolean(cx, option(1)),
        sputnik_is_null(cx, scale) ? std::nullopt : std::make_optional(number(cx, scale)),
        boolean(cx, option(2)));
    } else {
      const auto batch_first = boolean(cx, option(0));
      auto q = batch_first ? inputs[0].transpose(0, 1) : inputs[0];
      auto k = batch_first ? inputs[1].transpose(0, 1) : inputs[1];
      auto v = batch_first ? inputs[2].transpose(0, 1) : inputs[2];
      auto bias = parameters[4].defined()
        ? at::cat({parameters[4], parameters[5], parameters[6]}) : at::Tensor();
      if (mask.defined() && mask.scalar_type() == at::kBool)
        mask = at::zeros(mask.sizes(), q.options()).masked_fill(mask, -std::numeric_limits<double>::infinity());
      namespace F = torch::nn::functional;
      auto options = F::MultiheadAttentionForwardFuncOptions(q.size(-1), integer(cx, option(1)),
        {}, bias, parameters[8], parameters[9], boolean(cx, option(2)), 0,
        parameters[3], parameters[7]);
      options.training(false).need_weights(true).average_attn_weights(false)
        .attn_mask(mask).use_separate_proj_weight(true)
        .q_proj_weight(parameters[0]).k_proj_weight(parameters[1]).v_proj_weight(parameters[2]);
      std::tie(result, weights) = F::multi_head_attention_forward(q, k, v, options);
      if (batch_first) result = result.transpose(0, 1);
    }
    auto loss = result.square().sum();
    if (weights.defined()) loss = loss + weights.square().sum();
    loss.backward();
    auto optional = [&](const at::Tensor &value) {
      return value.defined() ? wrap(cx, value.detach()) : sputnik_make_null(cx);
    };
    std::vector<SputnikValue> input_gradients, parameter_gradients;
    for (const auto &input : inputs) input_gradients.push_back(optional(input.grad()));
    for (const auto &parameter : parameters)
      parameter_gradients.push_back(parameter.defined() ? optional(parameter.grad()) : sputnik_make_null(cx));
    const SputnikValue values[] = {optional(result), optional(weights),
      sputnik_make_list(cx, input_gradients.data(), input_gradients.size()),
      sputnik_make_list(cx, parameter_gradients.data(), parameter_gradients.size()),
      mask.defined() && mask.is_leaf() && mask.requires_grad() ? optional(mask.grad()) : sputnik_make_null(cx)};
    *out = sputnik_make_list(cx, values, 5);
  });
}
