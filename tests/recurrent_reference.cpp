#include "bridge.h"

// Test-only CPU oracle. Production recurrent gates and loops stay in Sputnik.
extern "C" SputnikStatus amt_recurrent_reference(SputnikCtx *cx, const SputnikValue *a,
                                                size_t n, SputnikValue *out) {
  return sputnik_torch::boundary(cx, [&] {
    using namespace sputnik_torch;
    require(n == 8 || n == 9, "ArgumentError", "expected dense or packed recurrent arguments");
    ExecutionGuard mode(0);
    auto leaf = [&](SputnikValue value) {
      return core(cx, value).tensor.detach().to(at::kCPU).clone().set_requires_grad(true);
    };
    const auto kind = string(cx, a[0]);
    auto input = leaf(a[1]);
    std::vector<at::Tensor> parameters, state;
    for (size_t i = 0; i < sputnik_list_len(cx, a[2]); ++i)
      parameters.push_back(leaf(sputnik_list_at(cx, a[2], i)));
    for (size_t i = 0; i < sputnik_list_len(cx, a[3]); ++i)
      state.push_back(leaf(sputnik_list_at(cx, a[3], i)));
    const auto bias = boolean(cx, a[4]);
    const auto layers = integer(cx, a[5]);
    const auto bidirectional = boolean(cx, a[6]);
    const auto batch_first = boolean(cx, a[7]);
    at::Tensor output, hidden, memory;
    if (n == 9) {
      const auto sizes = core(cx, a[8]).tensor.to(at::kCPU);
      if (kind == "lstm") {
        std::tie(output, hidden, memory) = at::lstm(input, sizes, state, parameters,
          bias, layers, 0, false, bidirectional);
      } else if (kind == "gru") {
        std::tie(output, hidden) = at::gru(input, sizes, state[0], parameters,
          bias, layers, 0, false, bidirectional);
      } else if (kind == "relu") {
        std::tie(output, hidden) = at::rnn_relu(input, sizes, state[0], parameters,
          bias, layers, 0, false, bidirectional);
      } else {
        std::tie(output, hidden) = at::rnn_tanh(input, sizes, state[0], parameters,
          bias, layers, 0, false, bidirectional);
      }
    } else if (kind == "lstm") {
      std::tie(output, hidden, memory) = at::lstm(input, state, parameters,
        bias, layers, 0, false, bidirectional, batch_first);
    } else if (kind == "gru") {
      std::tie(output, hidden) = at::gru(input, state[0], parameters,
        bias, layers, 0, false, bidirectional, batch_first);
    } else if (kind == "relu") {
      std::tie(output, hidden) = at::rnn_relu(input, state[0], parameters,
        bias, layers, 0, false, bidirectional, batch_first);
    } else {
      std::tie(output, hidden) = at::rnn_tanh(input, state[0], parameters,
        bias, layers, 0, false, bidirectional, batch_first);
    }
    auto loss = output.square().sum() + hidden.square().sum();
    if (memory.defined()) loss = loss + memory.square().sum();
    loss.backward();
    std::vector<SputnikValue> parameter_gradients;
    for (const auto &parameter : parameters) parameter_gradients.push_back(wrap(cx, parameter.grad()));
    const auto optional = [&](const at::Tensor &value) {
      return value.defined() ? wrap(cx, value.detach()) : sputnik_make_null(cx);
    };
    const SputnikValue values[] = {wrap(cx, output.detach()), wrap(cx, hidden.detach()), optional(memory),
      wrap(cx, input.grad()), wrap(cx, state[0].grad()),
      kind == "lstm" ? wrap(cx, state[1].grad()) : sputnik_make_null(cx),
      sputnik_make_list(cx, parameter_gradients.data(), parameter_gradients.size())};
    *out = sputnik_make_list(cx, values, 7);
  });
}

extern "C" SputnikStatus amt_sequence_reference(SputnikCtx *cx, const SputnikValue *a,
                                               size_t n, SputnikValue *out) {
  return sputnik_torch::boundary(cx, [&] {
    using namespace sputnik_torch;
    argc_check(n, 3);
    ExecutionGuard mode(0);
    std::vector<at::Tensor> inputs;
    for (size_t i = 0; i < sputnik_list_len(cx, a[1]); ++i) {
      auto input = core(cx, sputnik_list_at(cx, a[1], i)).tensor.detach().to(at::kCPU).clone();
      if (input.is_floating_point()) input.set_requires_grad(true);
      inputs.push_back(input);
    }
    auto option = [&](size_t i) { return sputnik_list_at(cx, a[2], i); };
    at::Tensor result, metadata;
    const auto op = string(cx, a[0]);
    if (op == "pack") {
      std::tie(result, metadata) = at::_pack_padded_sequence(inputs[0], inputs[1], boolean(cx, option(0)));
    } else if (op == "unpack") {
      std::tie(result, metadata) = at::_pad_packed_sequence(inputs[0], inputs[1], boolean(cx, option(0)),
        number(cx, option(1)), integer(cx, option(2)));
    } else {
      result = at::pad_sequence(inputs, boolean(cx, option(0)), number(cx, option(1)), string(cx, option(2)));
    }
    if (result.requires_grad()) result.square().sum().backward();
    std::vector<SputnikValue> gradients;
    for (const auto &input : inputs)
      gradients.push_back(input.grad().defined() ? wrap(cx, input.grad()) : sputnik_make_null(cx));
    const SputnikValue values[] = {wrap(cx, result.detach()),
      metadata.defined() ? wrap(cx, metadata) : sputnik_make_null(cx),
      sputnik_make_list(cx, gradients.data(), gradients.size())};
    *out = sputnik_make_list(cx, values, 3);
  });
}
