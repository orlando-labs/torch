#include "bridge.h"
#include <ATen/ops/_foreach_add.h>
#include <unordered_set>
#include <ATen/ops/_foreach_div.h>
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_foreach_pow.h>
#include <ATen/ops/_foreach_sqrt.h>
#include <ATen/ops/_foreach_sub.h>

using namespace sputniktorch;
namespace {
std::vector<at::Tensor> tensors(SputnikCtx *cx, SputnikValue list, bool mutating = false) {
  require(sputnik_is_list(cx, list), "TypeError", "expected tensor core list");
  std::vector<at::Tensor> values;
  const auto count = sputnik_list_len(cx, list);
  values.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    auto &value = core(cx, sputnik_list_at(cx, list, i));
    if (mutating) writable(value);
    values.push_back(value.tensor);
  }
  return values;
}
SputnikValue wrapped(SputnikCtx *cx, std::vector<at::Tensor> values) {
  std::vector<SputnikValue> result;
  result.reserve(values.size());
  for (auto &value : values) result.push_back(wrap(cx, std::move(value)));
  return sputnik_make_list(cx, result.data(), result.size());
}
template <class F>
void operand(SputnikCtx *cx, SputnikValue value, size_t count, F &&apply) {
  if (!sputnik_is_list(cx, value)) {
    apply(at::Scalar(number(cx, value)));
    return;
  }
  require(sputnik_list_len(cx, value) == count, "Sputniktorch.Errors.ShapeError",
          "tensor list operands must have equal lengths");
  if (count != 0 && sputnik_is_handle(cx, sputnik_list_at(cx, value, 0))) {
    apply(at::TensorList(tensors(cx, value)));
  } else {
    std::vector<at::Scalar> scalars;
    scalars.reserve(count);
    for (size_t i = 0; i < count; ++i)
      scalars.emplace_back(number(cx, sputnik_list_at(cx, value, i)));
    apply(at::ArrayRef<at::Scalar>(scalars));
  }
}
} // namespace

extern "C" {
SputnikStatus amt_foreach_binary(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]));
    auto values = tensors(cx, a[0]);
    auto op = string(cx, a[2]);
    std::vector<at::Tensor> result;
    operand(cx, a[1], values.size(), [&](const auto &rhs) {
      if (values.empty()) return;
      if (op == "add") result = at::_foreach_add(values, rhs);
      else if (op == "sub") result = at::_foreach_sub(values, rhs);
      else if (op == "mul") result = at::_foreach_mul(values, rhs);
      else if (op == "div") result = at::_foreach_div(values, rhs);
      else throw Error("ArgumentError", "unknown foreach binary operation");
    });
    *out = wrapped(cx, std::move(result));
  });
}
SputnikStatus amt_foreach_unary(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 3);
    ExecutionGuard mode(integer(cx, a[2]));
    auto values = tensors(cx, a[0]);
    auto op = string(cx, a[1]);
    std::vector<at::Tensor> result;
    if (op == "square") {
      if (!values.empty()) result = at::_foreach_pow(values, 2);
    } else if (op == "sqrt") {
      if (!values.empty()) result = at::_foreach_sqrt(values);
    } else throw Error("ArgumentError", "unknown foreach unary operation");
    *out = wrapped(cx, std::move(result));
  });
}
SputnikStatus amt_foreach_mutate(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]), true);
    auto values = tensors(cx, a[0], true);
    auto op = string(cx, a[2]);
    operand(cx, a[1], values.size(), [&](const auto &rhs) {
      if (values.empty()) return;
      if (op == "add") at::_foreach_add_(values, rhs);
      else if (op == "sub") at::_foreach_sub_(values, rhs);
      else if (op == "mul") at::_foreach_mul_(values, rhs);
      else throw Error("ArgumentError", "unknown foreach mutation");
    });
    *out = a[0];
  });
}
SputnikStatus amt_foreach_gradients(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 2);
    ExecutionGuard mode(integer(cx, a[1]));
    require(sputnik_is_list(cx, a[0]), "TypeError", "expected tensor core list");
    std::vector<SputnikValue> result;
    const auto count = sputnik_list_len(cx, a[0]);
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      auto &value = core(cx, sputnik_list_at(cx, a[0], i));
      result.push_back(value.tensor.grad().defined()
                           ? wrap(cx, value.tensor.grad(), value.readonly)
                           : sputnik_make_null(cx));
    }
    *out = sputnik_make_list(cx, result.data(), result.size());
  });
}
SputnikStatus amt_foreach_zero_grad(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 3);
    ExecutionGuard mode(integer(cx, a[2]), true);
    auto values = tensors(cx, a[0], true);
    if (boolean(cx, a[1])) {
      for (auto &value : values) value.mutable_grad() = at::Tensor();
    } else {
      at::NoGradGuard guard;
      for (auto &value : values)
        if (value.grad().defined()) value.mutable_grad().zero_();
    }
    *out = sputnik_make_null(cx);
  });
}
SputnikStatus amt_tensor_list_packable(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    auto values = tensors(cx, a[0]);
    bool compatible = !values.empty();
    for (const auto &value : values)
      compatible = compatible && value.layout() == at::kStrided &&
                   value.device() == values[0].device() &&
                   value.scalar_type() == values[0].scalar_type();
    *out = sputnik_make_bool(cx, compatible);
  });
}
SputnikStatus amt_tensor_list_independent(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    auto values = tensors(cx, a[0]);
    std::unordered_set<const c10::StorageImpl *> seen;
    bool independent = true;
    for (const auto &value : values) {
      if (value.layout() != at::kStrided || !seen.insert(value.storage().unsafeGetStorageImpl()).second) {
        independent = false;
        break;
      }
    }
    *out = sputnik_make_bool(cx, independent);
  });
}
SputnikStatus amt_tensor_list_index_select(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]));
    auto values = tensors(cx, a[0]);
    const auto dim = integer(cx, a[1]);
    const auto &indices = core(cx, a[2]).tensor;
    for (auto &value : values) value = value.index_select(dim, indices);
    *out = wrapped(cx, std::move(values));
  });
}
SputnikStatus amt_tensor_list_pack(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 2);
    ExecutionGuard mode(integer(cx, a[1]));
    auto values = tensors(cx, a[0]);
    require(!values.empty(), "ArgumentError", "pack requires a nonempty tensor list");
    for (auto &value : values) {
      require(value.layout() == at::kStrided && value.device() == values[0].device() &&
                  value.scalar_type() == values[0].scalar_type(),
              "Sputniktorch.Errors.DTypeError", "pack requires matching dense tensors");
    }
    for (auto &value : values) value = value.reshape({-1});
    *out = wrap(cx, at::cat(values));
  });
}
SputnikStatus amt_tensor_list_unpack(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 3);
    ExecutionGuard mode(integer(cx, a[2]));
    auto &flat = core(cx, a[0]);
    require(flat.tensor.dim() == 1, "Sputniktorch.Errors.ShapeError", "unpack requires a flat tensor");
    auto templates = tensors(cx, a[1]);
    int64_t total = 0;
    for (const auto &value : templates) {
      require(value.scalar_type() == flat.tensor.scalar_type() &&
                  value.device() == flat.tensor.device(),
              "Sputniktorch.Errors.DTypeError", "unpack requires matching templates");
      require(value.numel() <= flat.tensor.numel() - total,
              "Sputniktorch.Errors.ShapeError", "unpack template sizes exceed flat tensor");
      total += value.numel();
    }
    require(total == flat.tensor.numel(), "Sputniktorch.Errors.ShapeError", "unpack template sizes differ");
    std::vector<SputnikValue> result;
    result.reserve(templates.size());
    int64_t offset = 0;
    for (const auto &value : templates) {
      result.push_back(wrap(cx, flat.tensor.narrow(0, offset, value.numel()).view(value.sizes()), flat.readonly));
      offset += value.numel();
    }
    *out = sputnik_make_list(cx, result.data(), result.size());
  });
}
}
