#include "tensor/kernel_args.h"
#include <cmath>

namespace {
using namespace sputnik_torch;

at::Tensor interpolate(KernelArgs &k) {
  const auto x = k.tensor(0);
  const auto dimensions = x.dim() - 2;
  require(dimensions >= 1 && dimensions <= 3, "Torch.Errors.ShapeError",
          "interpolate requires rank 3, 4 or 5");
  auto output = k.optional_integers(0);
  const auto name = k.string(2);
  const bool align = sputnik_is_null(k.cx, k.arg(3)) ? false : k.boolean(3);
  const bool recompute = sputnik_is_null(k.cx, k.arg(4)) ? false : k.boolean(4);
  const bool antialias = k.boolean(5);
  std::vector<std::optional<double>> scales(dimensions);
  if (!sputnik_is_null(k.cx, k.arg(1))) {
    require(!output, "ArgumentError", "size and scale_factor are mutually exclusive");
    const auto values = k.arg(1);
    require(sputnik_is_list(k.cx, values) && sputnik_list_len(k.cx, values) == size_t(dimensions),
            "ArgumentError", "scale_factor must match the spatial rank");
    output.emplace();
    for (int64_t i = 0; i < dimensions; ++i) {
      const auto scale = number(k.cx, sputnik_list_at(k.cx, values, i));
      require(std::isfinite(scale) && scale > 0, "ArgumentError", "scale_factor must be finite and positive");
      const auto length = std::floor(double(x.size(i + 2)) * scale);
      require(length > 0 && length < std::ldexp(1.0, 63), "Torch.Errors.ShapeError",
              "scaled output size is empty or too large");
      output->push_back(int64_t(length));
      if (!recompute && name != "area") scales[i] = scale;
    }
  }
  require(output && output->size() == size_t(dimensions), "ArgumentError", "missing spatial output size");
  for (auto length : *output)
    require(length > 0, "Torch.Errors.ShapeError", "output sizes must be positive");
  if (name == "nearest") {
    if (dimensions == 1) return at::upsample_nearest1d(x, *output, scales[0]);
    if (dimensions == 2) return at::upsample_nearest2d(x, *output, scales[0], scales[1]);
    return at::upsample_nearest3d(x, *output, scales[0], scales[1], scales[2]);
  }
  if (name == "nearest-exact") {
    if (dimensions == 1) return at::_upsample_nearest_exact1d(x, *output, scales[0]);
    if (dimensions == 2) return at::_upsample_nearest_exact2d(x, *output, scales[0], scales[1]);
    return at::_upsample_nearest_exact3d(x, *output, scales[0], scales[1], scales[2]);
  }
  if (name == "area") {
    if (dimensions == 1) return at::adaptive_avg_pool1d(x, *output);
    if (dimensions == 2) return at::adaptive_avg_pool2d(x, *output);
    return at::adaptive_avg_pool3d(x, *output);
  }
  if (name == "linear" && dimensions == 1)
    return at::upsample_linear1d(x, *output, align, scales[0]);
  if (name == "bilinear" && dimensions == 2)
    return antialias ? at::_upsample_bilinear2d_aa(x, *output, align, scales[0], scales[1])
                     : at::upsample_bilinear2d(x, *output, align, scales[0], scales[1]);
  if (name == "bicubic" && dimensions == 2)
    return antialias ? at::_upsample_bicubic2d_aa(x, *output, align, scales[0], scales[1])
                     : at::upsample_bicubic2d(x, *output, align, scales[0], scales[1]);
  if (name == "trilinear" && dimensions == 3)
    return at::upsample_trilinear3d(x, *output, align, scales[0], scales[1], scales[2]);
  if (name == "lanczos" && dimensions == 2) {
    require(x.is_cpu(), "Torch.Errors.DeviceError", "lanczos interpolation supports CPU only");
    require(antialias, "ArgumentError", "lanczos requires antialias");
    return at::_upsample_lanczos2d_aa(x, *output, align, scales[0], scales[1]);
  }
  throw Error("ArgumentError", "interpolation mode does not match the spatial rank");
}

void verify_unpool(const at::Tensor &x, const at::Tensor &indices,
                   const std::vector<int64_t> &output) {
  require(indices.scalar_type() == at::kLong, "Torch.Errors.DTypeError",
          "unpool indices must be int64");
  require(indices.device() == x.device(), "Torch.Errors.DeviceError",
          "unpool indices must share the input device");
  require(indices.sizes() == x.sizes(), "Torch.Errors.ShapeError",
          "unpool input and indices must have the same shape");
  int64_t length = 1;
  for (auto axis : output) {
    require(axis > 0 && length <= std::numeric_limits<int64_t>::max() / axis,
            "Torch.Errors.ShapeError", "unpool output size is invalid");
    length *= axis;
  }
  // Reject bounds on the host before an asynchronous Metal kernel can fail.
  if (x.is_mps() && indices.numel() > 0)
    require(indices.min().item<int64_t>() >= 0 && indices.max().item<int64_t>() < length,
            "Torch.Errors.ShapeError", "unpool index is out of bounds");
}
} // namespace

extern "C" SputnikStatus amt_nn_spatial(SputnikCtx *cx, const SputnikValue *a,
                                       size_t n, SputnikValue *out) {
  return sputnik_torch::boundary(cx, [&] {
    using namespace sputnik_torch;
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]));
    KernelArgs k(cx, a[1], a[2]);
    const auto op = string(cx, a[0]);
    const auto x = k.tensor(0);
    at::Tensor result;
    if (op == "interpolate") result = interpolate(k);
    else if (op == "max_pool1d" || op == "max_pool2d" || op == "max_pool3d") {
      auto kernel = k.integers(0), stride = k.integers(1), padding = k.integers(2), dilation = k.integers(3);
      const auto ceil = k.boolean(4);
      if (k.boolean(5)) {
        auto pair = op == "max_pool1d" ? at::max_pool1d_with_indices(x, kernel, stride, padding, dilation, ceil)
                  : op == "max_pool2d" ? at::max_pool2d_with_indices(x, kernel, stride, padding, dilation, ceil)
                                       : at::max_pool3d_with_indices(x, kernel, stride, padding, dilation, ceil);
        *out = k.result(pair);
        return;
      }
      result = op == "max_pool1d" ? at::max_pool1d(x, kernel, stride, padding, dilation, ceil)
             : op == "max_pool2d" ? at::max_pool2d(x, kernel, stride, padding, dilation, ceil)
                                  : at::max_pool3d(x, kernel, stride, padding, dilation, ceil);
    } else if (op == "adaptive_max_pool1d" || op == "adaptive_max_pool2d" || op == "adaptive_max_pool3d") {
      auto pair = op == "adaptive_max_pool1d" ? at::adaptive_max_pool1d(x, k.integers(0))
                : op == "adaptive_max_pool2d" ? at::adaptive_max_pool2d(x, k.integers(0))
                                             : at::adaptive_max_pool3d(x, k.integers(0));
      if (k.boolean(1)) {
        *out = k.result(pair);
        return;
      }
      result = std::get<0>(pair);
    } else if (op == "avg_pool1d" || op == "avg_pool2d" || op == "avg_pool3d") {
      auto kernel = k.integers(0), stride = k.integers(1), padding = k.integers(2);
      result = op == "avg_pool1d" ? at::avg_pool1d(x, kernel, stride, padding, k.boolean(3), k.boolean(4))
             : op == "avg_pool2d" ? at::avg_pool2d(x, kernel, stride, padding, k.boolean(3), k.boolean(4))
                                  : at::avg_pool3d(x, kernel, stride, padding, k.boolean(3), k.boolean(4));
    } else if (op == "max_unpool1d" || op == "max_unpool2d" || op == "max_unpool3d") {
      const auto indices = k.tensor(1);
      auto output = k.integers(0);
      verify_unpool(x, indices, output);
      if (op == "max_unpool1d") {
        output.push_back(1);
        result = at::max_unpool2d(x.unsqueeze(-1), indices.unsqueeze(-1), output).squeeze(-1);
      } else if (op == "max_unpool2d") result = at::max_unpool2d(x, indices, output);
      else result = at::max_unpool3d(x, indices, output, k.integers(1), k.integers(2));
    } else throw Error("ArgumentError", "unknown spatial kernel");
    *out = k.result(std::move(result));
  });
}
