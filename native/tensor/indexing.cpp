#include "tensor/kernel_args.h"
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorIndexing.h>
#include <ATen/WrapDimUtils.h>

namespace {
using namespace sputnik_torch;

void check_bounds(const at::Tensor &index, int64_t length, bool negative) {
  if (index.numel() == 0 ||
      (index.scalar_type() != at::kLong && index.scalar_type() != at::kInt)) return;
  const auto low = index.min().item<int64_t>();
  const auto high = index.max().item<int64_t>();
  require(low >= (negative ? -length : 0) && high < length,
          "Torch.Errors.ShapeError", "index is out of bounds");
}

int64_t axis_size(const at::Tensor &x, int64_t dim) {
  const auto axis = at::maybe_wrap_dim(dim, x.dim());
  return x.dim() == 0 ? 1 : x.size(axis);
}

void check_advanced_bounds(const at::Tensor &x,
                          const std::vector<at::indexing::TensorIndex> &values) {
  if (!x.is_mps()) return;
  auto consumed = [](const auto &value) -> int64_t {
    if (value.is_integer() || value.is_slice()) return 1;
    if (!value.is_tensor()) return 0;
    const auto &tensor = value.tensor();
    return tensor.scalar_type() == at::kBool || tensor.scalar_type() == at::kByte
               ? tensor.dim() : 1;
  };
  int64_t total = 0, axis = 0;
  for (const auto &value : values) total += consumed(value);
  if (total > x.dim()) return; // ATen reports invalid dimensionality before dispatch.
  for (const auto &value : values) {
    if (value.is_ellipsis()) axis += x.dim() - total;
    else {
      if (value.is_tensor() && axis < x.dim())
        check_bounds(value.tensor(), x.size(axis), true);
      axis += consumed(value);
    }
  }
}

std::vector<at::indexing::TensorIndex> indices(SputnikCtx *cx, SputnikValue values) {
  require(sputnik_is_list(cx, values), "TypeError", "indices must be an Array");
  std::vector<at::indexing::TensorIndex> result;
  result.reserve(sputnik_list_len(cx, values));
  for (size_t i = 0; i < sputnik_list_len(cx, values); ++i) {
    auto value = sputnik_list_at(cx, values, i);
    if (sputnik_is_null(cx, value)) result.emplace_back(at::indexing::None);
    else if (sputnik_is_bool(cx, value)) result.emplace_back(boolean(cx, value));
    else if (sputnik_is_int(cx, value)) result.emplace_back(integer(cx, value));
    else if (sputnik_is_handle(cx, value)) result.emplace_back(core(cx, value).tensor);
    else if (sputnik_is_str(cx, value)) {
      require(string(cx, value) == "...", "TypeError", "unknown tensor index");
      result.emplace_back(at::indexing::Ellipsis);
    } else if (sputnik_is_list(cx, value)) {
      require(sputnik_list_len(cx, value) == 3, "TypeError", "invalid Slice");
      auto bound = [&](size_t j) -> std::optional<c10::SymInt> {
        auto v = sputnik_list_at(cx, value, j);
        return sputnik_is_null(cx, v) ? std::nullopt
                                     : std::optional<c10::SymInt>(integer(cx, v));
      };
      result.emplace_back(at::indexing::Slice(bound(0), bound(1), bound(2)));
    } else throw Error("TypeError", "expected Int, Bool, Tensor, Slice, null or :ellipsis");
  }
  return result;
}

void check_index_overlap(const at::Tensor &target,
                         const std::vector<at::indexing::TensorIndex> &values) {
  for (const auto &value : values)
    if (value.is_tensor()) at::assert_no_overlap(target, value.tensor());
}

void index_put(SputnikCtx *cx, at::Tensor &target,
               const std::vector<at::indexing::TensorIndex> &index,
               SputnikValue value, bool accumulate) {
  if (accumulate) {
    // The accumulating ATen operator accepts tensor indices, not view slices.
    c10::List<std::optional<at::Tensor>> tensors;
    for (const auto &item : index) {
      require(item.is_tensor(), "TypeError",
              "accumulating index_put requires Tensor or Array indices");
      tensors.push_back(item.tensor());
    }
    auto source = sputnik_is_handle(cx, value)
                      ? core(cx, value).tensor
                      : at::scalar_tensor(numeric_scalar(cx, value), target.options());
    target.index_put_(tensors, source, true);
  } else if (sputnik_is_handle(cx, value)) {
    target.index_put_(index, core(cx, value).tensor);
  } else {
    target.index_put_(index, numeric_scalar(cx, value));
  }
}
} // namespace

extern "C" SputnikStatus amt_indexing(SputnikCtx *cx, const SputnikValue *a,
                                     size_t n, SputnikValue *out) {
  return sputnik_torch::boundary(cx, [&] {
    using namespace sputnik_torch;
    argc_check(n, 5);
    const auto op = string(cx, a[0]);
    KernelArgs k(cx, a[1], a[2]);
    const bool mutating = boolean(cx, a[3]);
    const bool supports_mutation = op == "index_put" || op == "put" || op == "scatter" ||
        op == "scatter_add" || op == "scatter_reduce" || op == "index_add" ||
        op == "index_copy" || op == "index_fill" || op == "index_reduce" ||
        op == "masked_fill" || op == "masked_scatter" || op == "put_prepare";
    require(!mutating || supports_mutation, "ArgumentError", "operation has no bang variant");
    ExecutionGuard mode(integer(cx, a[4]), mutating);
    auto &receiver = core(cx, k.input(0));
    auto &x = receiver.tensor;
    if (mutating) {
      writable(receiver);
      at::assert_no_internal_overlap(x);
      for (size_t i = 1; i < sputnik_list_len(cx, k.inputs); ++i)
        if (sputnik_is_handle(cx, k.input(i)))
          at::assert_no_overlap(x, k.tensor(i));
    }
    if (op == "put_prepare") {
      const auto index = k.tensor(1);
      const auto source = k.tensor(2);
      require(index.scalar_type() == at::kLong, "Torch.Errors.DTypeError",
              "put requires int64 indices");
      require(source.scalar_type() == x.scalar_type(), "Torch.Errors.DTypeError",
              "put source must have the receiver dtype");
      require(source.device() == x.device(), "Torch.Errors.DeviceError",
              "put source must have the receiver device");
      require(source.numel() == index.numel(), "Torch.Errors.ShapeError",
              "put source and index must have the same number of elements");
      check_bounds(index, x.numel(), true);
      *out = k.input(0);
      return;
    }
    // A Metal bounds assertion is asynchronous and can poison the command
    // stream. Validate index tensors before submitting a potentially bad write.
    if (x.is_mps()) {
      if (op == "gather" || op == "scatter" || op == "scatter_add" ||
          op == "scatter_reduce" || op == "index_add" || op == "index_copy" ||
          op == "index_fill" || op == "index_reduce") {
        check_bounds(k.tensor(1), axis_size(x, k.integer(0)), op == "index_fill");
      } else if (op == "take_along_dim") {
        const auto dim = k.optional_integer(0);
        check_bounds(k.tensor(1), dim ? axis_size(x, *dim) : x.numel(), false);
      }
    }

    at::Tensor result;
    if (op == "index" || op == "index_put") {
      auto index = indices(cx, k.arg(0));
      check_advanced_bounds(x, index);
      if (mutating) check_index_overlap(x, index);
      if (op == "index") {
        require(!mutating, "ArgumentError", "index is read-only");
        result = x.index(index);
      } else {
        result = mutating ? x : x.clone();
        index_put(cx, result, index, k.input(1), k.boolean(1));
      }
    } else if (op == "gather") {
      result = x.gather(k.integer(0), k.tensor(1), k.boolean(1));
    } else if (op == "take") {
      result = x.take(k.tensor(1));
    } else if (op == "take_along_dim") {
      result = x.take_along_dim(k.tensor(1), k.optional_integer(0));
    } else if (op == "put") {
      result = mutating ? x.put_(k.tensor(1), k.tensor(2), k.boolean(0))
                        : x.put(k.tensor(1), k.tensor(2), k.boolean(0));
    } else if (op == "scatter") {
      const auto dim = k.integer(0);
      const auto index = k.tensor(1);
      const bool reduce = !sputnik_is_null(cx, k.arg(1));
      if (sputnik_is_handle(cx, k.input(2))) {
        const auto source = k.tensor(2);
        if (reduce)
          result = mutating ? x.scatter_(dim, index, source, k.string(1))
                            : x.scatter(dim, index, source, k.string(1));
        else result = mutating ? x.scatter_(dim, index, source) : x.scatter(dim, index, source);
      } else {
        const auto value = numeric_scalar(cx, k.input(2));
        if (reduce)
          result = mutating ? x.scatter_(dim, index, value, k.string(1))
                            : x.scatter(dim, index, value, k.string(1));
        else result = mutating ? x.scatter_(dim, index, value) : x.scatter(dim, index, value);
      }
    } else if (op == "scatter_add") {
      result = mutating ? x.scatter_add_(k.integer(0), k.tensor(1), k.tensor(2))
                        : x.scatter_add(k.integer(0), k.tensor(1), k.tensor(2));
    } else if (op == "scatter_reduce") {
      result = mutating ? x.scatter_reduce_(k.integer(0), k.tensor(1), k.tensor(2),
                                            k.string(1), k.boolean(2))
                        : x.scatter_reduce(k.integer(0), k.tensor(1), k.tensor(2),
                                           k.string(1), k.boolean(2));
    } else if (op == "index_add") {
      const auto alpha = numeric_scalar(cx, k.arg(1));
      result = mutating ? x.index_add_(k.integer(0), k.tensor(1), k.tensor(2), alpha)
                        : x.index_add(k.integer(0), k.tensor(1), k.tensor(2), alpha);
    } else if (op == "index_copy") {
      result = mutating ? x.index_copy_(k.integer(0), k.tensor(1), k.tensor(2))
                        : x.index_copy(k.integer(0), k.tensor(1), k.tensor(2));
    } else if (op == "index_fill") {
      if (sputnik_is_handle(cx, k.input(2)))
        result = mutating ? x.index_fill_(k.integer(0), k.tensor(1), k.tensor(2))
                          : x.index_fill(k.integer(0), k.tensor(1), k.tensor(2));
      else {
        const auto value = numeric_scalar(cx, k.input(2));
        result = mutating ? x.index_fill_(k.integer(0), k.tensor(1), value)
                          : x.index_fill(k.integer(0), k.tensor(1), value);
      }
    } else if (op == "index_reduce") {
      result = mutating ? x.index_reduce_(k.integer(0), k.tensor(1), k.tensor(2),
                                          k.string(1), k.boolean(2))
                        : x.index_reduce(k.integer(0), k.tensor(1), k.tensor(2),
                                         k.string(1), k.boolean(2));
    } else if (op == "masked_select") {
      result = x.masked_select(k.tensor(1));
    } else if (op == "masked_fill") {
      if (sputnik_is_handle(cx, k.input(2)))
        result = mutating ? x.masked_fill_(k.tensor(1), k.tensor(2))
                          : x.masked_fill(k.tensor(1), k.tensor(2));
      else {
        const auto value = numeric_scalar(cx, k.input(2));
        result = mutating ? x.masked_fill_(k.tensor(1), value) : x.masked_fill(k.tensor(1), value);
      }
    } else if (op == "masked_scatter") {
      result = mutating ? x.masked_scatter_(k.tensor(1), k.tensor(2))
                        : x.masked_scatter(k.tensor(1), k.tensor(2));
    } else if (op == "nonzero") {
      if (k.boolean(0)) {
        std::vector<SputnikValue> values;
        for (auto &tensor : x.nonzero_numpy()) values.push_back(k.result(std::move(tensor)));
        *out = sputnik_make_list(cx, values.data(), values.size());
        return;
      }
      result = x.nonzero();
    } else if (op == "select_scatter") {
      result = x.select_scatter(k.tensor(1), k.integer(0), k.integer(1));
    } else if (op == "slice_scatter") {
      result = x.slice_scatter(k.tensor(1), k.integer(0), k.optional_integer(1),
                               k.optional_integer(2), k.integer(3));
    } else if (op == "diagonal_scatter") {
      result = x.diagonal_scatter(k.tensor(1), k.integer(0), k.integer(1), k.integer(2));
    } else throw Error("ArgumentError", "unknown indexing operation");
    *out = mutating ? k.input(0) : k.result(std::move(result));
  });
}
