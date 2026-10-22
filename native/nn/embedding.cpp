#include "tensor/kernel_args.h"

namespace {
using namespace sputnik_torch;

void validate(const at::Tensor &indices, const at::Tensor &weight) {
  require(weight.dim() == 2 && weight.is_floating_point() && weight.layout() == at::kStrided,
          "Torch.Errors.ShapeError", "embedding weight must be a dense floating matrix");
  require(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
          "Torch.Errors.DTypeError", "embedding indices must be int32 or int64");
  require(indices.layout() == at::kStrided && indices.device() == weight.device(),
          "Torch.Errors.DeviceError", "embedding indices must be dense and share the weight device");
  // Reject invalid labels before a GPU gather or frequency-count backward.
  if (indices.numel() > 0)
    require(indices.min().item<int64_t>() >= 0 && indices.max().item<int64_t>() < weight.size(0),
            "Torch.Errors.ShapeError", "embedding index is out of bounds");
}

void gradients(const at::Tensor &weight, bool frequency, bool sparse) {
  require(!sparse || !frequency, "ArgumentError", "sparse embedding gradients do not support frequency scaling");
  require(!sparse || weight.is_cpu(), "Torch.Errors.DeviceError",
          "sparse embedding gradients are supported only on CPU");
}

void validate_bags(const at::Tensor &indices, const at::Tensor &weight,
                   const at::Tensor &offsets, const std::optional<at::Tensor> &samples,
                   bool include_last) {
  require(indices.dim() == 1 && offsets.dim() == 1, "Torch.Errors.ShapeError",
          "embedding bag indices and offsets must be 1D");
  require(offsets.scalar_type() == indices.scalar_type() && offsets.device() == indices.device() &&
              offsets.layout() == at::kStrided,
          "Torch.Errors.DTypeError", "embedding offsets must match index dtype/device");
  if (samples) {
    require(samples->sizes() == indices.sizes() && samples->scalar_type() == weight.scalar_type() &&
                samples->device() == weight.device() && samples->layout() == at::kStrided,
            "Torch.Errors.ShapeError", "per-sample weights must match index shape and weight dtype/device");
  }
  require(offsets.numel() > 0 || (!include_last && indices.numel() == 0),
          "Torch.Errors.ShapeError", "embedding bag offsets are missing");
  if (offsets.numel() == 0) return;
  const auto host = offsets.to(at::kCPU).to(at::kLong).contiguous();
  const auto data = host.data_ptr<int64_t>();
  require(data[0] == 0, "Torch.Errors.ShapeError", "first embedding bag offset must be zero");
  for (int64_t i = 0; i < host.numel(); ++i)
    require(data[i] >= 0 && data[i] <= indices.numel() && (i == 0 || data[i] >= data[i - 1]),
            "Torch.Errors.ShapeError", "embedding bag offsets must be sorted and in bounds");
  require(!include_last || data[host.numel() - 1] == indices.numel(),
          "Torch.Errors.ShapeError", "last embedding bag offset must equal the input length");
}
} // namespace

extern "C" SputnikStatus amt_embedding(SputnikCtx *cx, const SputnikValue *a,
                                      size_t n, SputnikValue *out) {
  return sputnik_torch::boundary(cx, [&] {
    using namespace sputnik_torch;
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]));
    const auto op = string(cx, a[0]);
    KernelArgs k(cx, a[1], a[2]);
    if (op == "where") {
      *out = k.result(at::where(k.tensor(0), k.tensor(1), k.tensor(2)));
      return;
    }
    const auto indices = k.tensor(0), weight = k.tensor(1);
    validate(indices, weight);
    if (op == "validate") { *out = sputnik_make_null(cx); return; }
    if (op == "embedding" || op == "embedding_validate") {
      gradients(weight, k.boolean(1), k.boolean(2));
      if (op == "embedding_validate") { *out = sputnik_make_null(cx); return; }
      *out = k.result(at::embedding(weight, indices, k.optional_integer(0).value_or(-1),
                                   k.boolean(1), k.boolean(2)));
    } else if (op == "bag" || op == "bag_validate") {
      gradients(weight, k.boolean(1), k.boolean(2));
      const auto offsets = k.tensor(2);
      const auto samples = sputnik_is_null(cx, k.input(3)) ? std::nullopt
                          : std::optional<at::Tensor>(k.tensor(3));
      validate_bags(indices, weight, offsets, samples, k.boolean(3));
      if (op == "bag_validate") { *out = sputnik_make_null(cx); return; }
      *out = k.result(std::get<0>(at::embedding_bag(weight, indices, offsets, k.boolean(1),
                              k.integer(0), k.boolean(2), samples, k.boolean(3), k.optional_integer(4))));
    } else
      throw Error("ArgumentError", "unknown embedding operation");
  });
}
