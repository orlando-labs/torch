#include "bridge.h"

namespace {
using namespace sputnik_torch;

std::vector<int64_t> lengths(SputnikCtx *cx, SputnikValue value) {
  if (sputnik_is_list(cx, value)) return integers(cx, value);
  const auto &tensor = core(cx, value).tensor;
  require(tensor.dim() <= 1 && (tensor.scalar_type() == at::kLong || tensor.scalar_type() == at::kInt),
          "Torch.Errors.DTypeError", "CTC lengths must be scalar/1D integer metadata");
  const auto host = tensor.to(at::kCPU).to(at::kLong).contiguous();
  if (host.numel() == 0) return {};
  const auto data = host.data_ptr<int64_t>();
  return std::vector<int64_t>(data, data + host.numel());
}
} // namespace

extern "C" SputnikStatus amt_ctc_loss(SputnikCtx *cx, const SputnikValue *a,
                                     size_t n, SputnikValue *out) {
  return sputnik_torch::boundary(cx, [&] {
    using namespace sputnik_torch;
    argc_check(n, 8);
    ExecutionGuard mode(integer(cx, a[7]));
    const auto &log_probs = core(cx, a[0]).tensor;
    const auto &targets = core(cx, a[1]).tensor;
    require(log_probs.dim() == 2 || log_probs.dim() == 3, "Torch.Errors.ShapeError",
            "CTC log probabilities must have shape [T, C] or [T, N, C]");
    require(targets.dim() == 1 || targets.dim() == 2, "Torch.Errors.ShapeError",
            "CTC targets must be concatenated or padded");
    require(targets.scalar_type() == at::kLong || targets.scalar_type() == at::kInt,
            "Torch.Errors.DTypeError", "CTC targets must be int32 or int64");
    const auto batch = log_probs.dim() == 2 ? 1 : log_probs.size(1);
    const auto classes = log_probs.size(-1), time = log_probs.size(0);
    const auto blank = integer(cx, a[4]);
    require(blank >= 0 && blank < classes, "ArgumentError", "CTC blank is out of bounds");
    const auto input_lengths = lengths(cx, a[2]);
    const auto target_lengths = lengths(cx, a[3]);
    require(input_lengths.size() == size_t(batch) && target_lengths.size() == size_t(batch),
            "Torch.Errors.ShapeError", "CTC lengths must match the batch");
    require(targets.dim() != 2 || targets.size(0) == batch, "Torch.Errors.ShapeError",
            "CTC padded targets must match the batch");
    // Inspect only active labels. Padded trailing entries may be sentinels.
    const auto host = targets.to(at::kCPU).to(at::kLong).contiguous();
    const auto labels = host.data_ptr<int64_t>();
    int64_t offset = 0;
    for (int64_t i = 0; i < batch; ++i) {
      require(input_lengths[i] >= 0 && input_lengths[i] <= time && target_lengths[i] >= 0,
              "Torch.Errors.ShapeError", "CTC length is out of bounds");
      if (targets.dim() == 2) {
        require(target_lengths[i] <= targets.size(1), "Torch.Errors.ShapeError",
                "CTC target length exceeds padded width");
        offset = i * targets.size(1);
      } else {
        require(target_lengths[i] <= targets.numel() - offset, "Torch.Errors.ShapeError",
                "CTC target lengths exceed concatenated labels");
      }
      for (int64_t j = 0; j < target_lengths[i]; ++j)
        require(labels[offset + j] >= 0 && labels[offset + j] < classes && labels[offset + j] != blank,
                "Torch.Errors.ShapeError", "CTC active label is invalid or blank");
      offset += target_lengths[i];
    }
    require(targets.dim() != 1 || offset == targets.numel(), "Torch.Errors.ShapeError",
            "CTC concatenated target length mismatch");
    *out = wrap(cx, at::ctc_loss(log_probs, targets, input_lengths, target_lengths,
                                blank, integer(cx, a[5]), boolean(cx, a[6])));
  });
}
