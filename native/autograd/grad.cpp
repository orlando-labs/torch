#include "bridge.h"
#include <torch/csrc/autograd/autograd.h>
using namespace sputniktorch;

extern "C" SputnikStatus amt_grad(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 7);
    require(integer(cx, a[6]) < 4, "Sputniktorch.Errors.ONNXError", "autograd.grad is unsupported during ONNX export");
    auto tensors = [&](SputnikValue list, bool optional) {
      require(sputnik_is_list(cx, list), "TypeError", "grad requires tensor lists");
      std::vector<at::Tensor> result;
      for (size_t i = 0; i < sputnik_list_len(cx, list); ++i) {
        auto value = sputnik_list_at(cx, list, i);
        result.push_back(optional && sputnik_is_null(cx, value) ? at::Tensor()
                                                              : core(cx, value).tensor);
      }
      return result;
    };
    auto outputs = tensors(a[0], false);
    auto inputs = tensors(a[1], false);
    require(!outputs.empty() && !inputs.empty(), "ArgumentError",
            "grad requires outputs and inputs");
    auto gradients = sputnik_is_null(cx, a[2]) ? std::vector<at::Tensor>() : tensors(a[2], true);
    auto result = torch::autograd::grad(outputs, inputs, gradients, boolean(cx, a[3]),
                                        boolean(cx, a[4]), boolean(cx, a[5]));
    std::vector<SputnikValue> values;
    for (auto &tensor : result)
      values.push_back(tensor.defined() ? wrap(cx, std::move(tensor)) : sputnik_make_null(cx));
    *out = sputnik_make_list(cx, values.data(), values.size());
  });
}
