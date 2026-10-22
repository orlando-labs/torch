package torch.spatial_test

from torch import tensor, zeros, ones, full, arange
from torch.nn import Upsample, UpsamplingNearest2d, UpsamplingBilinear2d
from torch.nn import MaxPool1d, MaxPool2d, MaxPool3d, MaxUnpool1d, MaxUnpool2d, MaxUnpool3d
from torch.nn import AdaptiveMaxPool1d, AdaptiveMaxPool2d, AdaptiveMaxPool3d
from torch.nn import LPPool1d, LPPool2d, LPPool3d
from torch.checks import check
import torch.nn.functional as f

export main, run

def close(x, expected, message, tolerance: 3e-4):
  check(x.shape == expected.shape and (x - expected).abs.sum.item < tolerance, message)

def rejected(&block):
  failed = false
  try:
    block()
  rescue ValueError:
    failed = true
  rescue TypeError:
    failed = true
  rescue ArgumentError:
    failed = true
  rescue Torch.Errors.Error:
    failed = true
  check(failed, "invalid spatial operation must raise")

def run(device):
  line = tensor([[[0.0, 10.0, 20.0]]], device:)
  close(f.interpolate(line, size: 5), tensor([[[0.0, 0.0, 10.0, 10.0, 20.0]]], device:), "nearest legacy coordinates")
  close(f.interpolate(line, size: 5, mode: "nearest-exact"),
    tensor([[[0.0, 0.0, 10.0, 20.0, 20.0]]], device:), "nearest exact coordinates")
  close(f.interpolate(line, size: 5, mode: :linear), tensor([[[0.0, 4.0, 10.0, 16.0, 20.0]]], device:), "linear half-pixel coordinates")
  close(f.interpolate(line, size: 5, mode: :linear, align_corners: true),
    tensor([[[0.0, 5.0, 10.0, 15.0, 20.0]]], device:), "aligned linear coordinates")
  close(f.interpolate(line, scale_factor: 1.5, mode: :linear),
    tensor([[[0.0, 5.0, 35.0 / 3, 55.0 / 3]]], device:), "explicit fractional scale")
  close(f.interpolate(line, scale_factor: [1.5], mode: :linear, recompute_scale_factor: true),
    tensor([[[0.0, 6.25, 13.75, 20.0]]], device:), "recomputed fractional scale")
  area_input = if device == "cpu" then line else tensor([[[0.0, 10.0, 10.0, 20.0]]], device:)
  close(f.interpolate(area_input, size: [2], mode: :area), tensor([[[5.0, 15.0]]], device:), "area downsample")
  if device == "mps":
    rejected(): f.interpolate(line, size: 2, mode: :area)
  input = line.copy.requires_grad!(true)
  f.interpolate(input, size: 5, mode: :linear, align_corners: true).sum.backward!
  close(input.grad, tensor([[[1.5, 2.0, 1.5]]], device:), "linear backward weights")

  image = tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device:)
  expected = tensor([[[[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [3.0, 3.5, 4.0]]]], device:)
  close(Upsample(size: [3, 3], mode: :bilinear, align_corners: true)(image), expected, "Upsample module")
  close(UpsamplingBilinear2d(size: (3, 3))(image), expected, "legacy bilinear module aligns corners")
  close(UpsamplingNearest2d(scale_factor: 2)(image), f.interpolate(image, size: 4), "legacy nearest module")
  close(f.interpolate(image, size: 3, mode: "nearest-exact"),
    tensor([[[[1.0, 2.0, 2.0], [3.0, 4.0, 4.0], [3.0, 4.0, 4.0]]]], device:), "2D exact nearest")
  close(f.interpolate(image, size: 2, mode: :bicubic), image, "bicubic identity coordinates")
  check(f.interpolate(image, scale_factor: [2, 3]).shape == [1, 1, 4, 6], "per-axis scale factors")
  close(f.interpolate(image, size: [1, 1], mode: :area), full([1, 1, 1, 1], 2.5, device:), "2D area")
  [:bilinear, :bicubic].each |mode|:
    input = ones([1, 1, 4, 4], device:, requires_grad: true)
    output = f.interpolate(input, size: [2, 2], mode:, antialias: true)
    close(output, ones([1, 1, 2, 2], device:), "antialias constant preservation")
    output.sum.backward!
    check((input.grad.sum.item - 4).abs < 1e-4, "antialias gradient sum")
    check(output.device.type == device and input.grad.device.type == device, "antialias backend")
    close(f.interpolate(input, size: [3, 3], mode:), ones([1, 1, 3, 3], device:), "2D interpolation constants")
  edge = tensor([[[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]], device:)
  close(f.interpolate(edge, size: 2, mode: :bilinear, antialias: true),
    tensor([[[[0.0, 3.0 / 7], [0.0, 3.0 / 7]]]], device:), "antialias filter edge reference")
  close(f.interpolate(edge, size: 2, mode: :bilinear),
    tensor([[[[0.0, 0.5], [0.0, 0.5]]]], device:), "non-antialias reference")
  frozen = image.snapshot
  output = f.interpolate(frozen, size: 2)
  check(not output.readonly?, "identity-size interpolation still owns storage")
  output.zero!
  close(frozen, image, "resize cannot mutate snapshot")
  close(f.interpolate(image.transpose(-1, -2), size: 3, mode: :bilinear, align_corners: true),
    expected.transpose(-1, -2), "noncontiguous resize")

  volume = arange(9, start: 1, dtype: "float32", device:).reshape([1, 1, 2, 2, 2])
  close(f.interpolate(volume, size: 1, mode: :area), full([1, 1, 1, 1, 1], 4.5, device:), "3D area")
  close(f.interpolate(volume, size: 3, mode: :trilinear, align_corners: true)[(0, 0, 1, 1, 1)],
    tensor(4.5, device:), "trilinear center")
  [:nearest, "nearest-exact", :trilinear].each |mode|:
    input = ones([1, 1, 2, 2, 2], device:, requires_grad: true)
    output = f.interpolate(input, scale_factor: 2, mode:)
    close(output, ones([1, 1, 4, 4, 4], device:), "3D interpolation constants")
    output.sum.backward!
    close(input.grad, full([1, 1, 2, 2, 2], 8, device:), "3D interpolation backward")
    check(output.device.type == device and input.grad.device.type == device, "3D resize backend")
  if device == "cpu":
    input = full([1, 1, 4, 4], 2.5, device:, requires_grad: true)
    output = f.interpolate(input, size: [2, 3], mode: :lanczos, antialias: true)
    close(output, full([1, 1, 2, 3], 2.5, device:), "Lanczos constant preservation")
    output.sum.backward!
    check((input.grad.sum.item - 6).abs < 1e-4, "Lanczos backward")
    close(f.interpolate(input, size: [2, 3], mode: :lanczos, antialias: true, align_corners: true),
      full([1, 1, 2, 3], 2.5, device:), "Lanczos aligned-corner kernel")
    double = image.to(device:, dtype: "float64")
    check(f.interpolate(double, size: 3, mode: :bicubic).dtype.to_str == "float64", "resize dtype")
  else:
    rejected(): f.interpolate(image, size: 3, mode: :lanczos, antialias: true)
  rejected(): f.interpolate(line)
  rejected(): f.interpolate(line, size: 3, scale_factor: 2)
  rejected(): f.interpolate(line, scale_factor: 0)
  rejected(): f.interpolate(line, scale_factor: -1)
  rejected(): f.interpolate(line, scale_factor: 0.01)
  rejected(): f.interpolate(line, scale_factor: "2")
  rejected(): f.interpolate(line, size: [3, 3])
  rejected(): f.interpolate(image, size: 3.5)
  rejected(): f.interpolate(image, size: 0)
  rejected(): f.interpolate(image, scale_factor: [2])
  rejected(): f.interpolate(image, size: 3, mode: :linear)
  rejected(): f.interpolate(line, size: 3, align_corners: false)
  rejected(): f.interpolate(image, size: 3, mode: "nearest-exact", align_corners: true)
  rejected(): f.interpolate(image, size: 3, recompute_scale_factor: true)
  rejected(): f.interpolate(volume, size: 3, mode: :trilinear, antialias: true)
  rejected(): f.interpolate(image, size: 3, mode: :lanczos)
  rejected(): f.interpolate(image, size: 3, mode: :unsupported)

  pools = [MaxPool1d(2, return_indices: true), MaxPool2d(2, return_indices: true), MaxPool3d(2, return_indices: true)]
  unpools = [MaxUnpool1d(2), MaxUnpool2d(2), MaxUnpool3d(2)]
  inputs = [tensor([[[1.0, 3.0, 2.0, 4.0]]], device:),
    tensor([[[[1.0, 3.0, 2.0, 4.0], [5.0, 7.0, 6.0, 8.0]]]], device:), volume]
  expected = [tensor([[[0.0, 3.0, 0.0, 4.0]]], device:),
    tensor([[[[0.0, 0.0, 0.0, 0.0], [0.0, 7.0, 0.0, 8.0]]]], device:),
    tensor([[[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 8.0]]]]], device:)]
  expected_indices = [tensor([[[1, 3]]], device:), tensor([[[[5, 7]]]], device:), tensor([[[[[7]]]]], device:)]
  3.times |i|:
    input = inputs[i].copy.requires_grad!(true)
    pair = pools[i](input)
    check(Tuple === pair, "pool returns a Tuple")
    (values, indices) = pair
    close(indices.to(device:, dtype: "float32"), expected_indices[i].to(device:, dtype: "float32"), "pool flattened indices")
    check(indices.dtype.to_str == "int64" and not indices.requires_grad?, "pool index dtype/autograd")
    output = unpools[i](values, indices)
    close(output, expected[i], "unpool sparse reconstruction")
    output.sum.backward!
    close(input.grad, expected[i].gt(0).to(device:, dtype: "float32"), "pool/unpool chain backward")
    close(unpools[i](values[0], indices[0]), expected[i][0], "unbatched unpool")
    check(output.device.type == device and indices.device.type == device and input.grad.device.type == device,
          "pool/unpool backend")

  pool_pair = f.max_pool1d(inputs[0], 2, return_indices: true)
  (values, indices) = pool_pair
  close(f.max_unpool1d(values, indices, 2), expected[0], "functional 1D pool/unpool")
  close(f.max_pool1d(inputs[0], 2, dilation: 2), tensor([[[2.0]]], device:), "pool dilation")
  padded = f.max_pool1d(inputs[0], 2, stride: 1, padding: 1, return_indices: true)
  close(padded[0], tensor([[[1.0, 3.0, 3.0, 4.0, 4.0]]], device:), "pool padding and stride")
  check(padded[1].to_a == [[[0, 1, 1, 3, 3]]], "pool padded indices")

  (values, indices) = f.max_pool2d(inputs[1], 2, return_indices: true)
  close(f.max_unpool2d(values, indices, 2), expected[1], "functional 2D pool/unpool")
  close(f.max_pool2d(inputs[1], 2), values, "pool default Tensor result")

  (values, indices) = f.max_pool3d(volume, 2, return_indices: true)
  close(f.max_unpool3d(values, indices, 2), expected[2], "functional 3D pool/unpool")
  close(f.max_pool3d(volume, 2), values, "functional 3D pool without indices")
  adaptive = [AdaptiveMaxPool1d(1, return_indices: true), AdaptiveMaxPool2d(1, return_indices: true),
    AdaptiveMaxPool3d(1, return_indices: true)]
  3.times |i|:
    if device == "mps" and i == 2:
      rejected(): adaptive[i](inputs[i])
    else:
      (values, indices) = adaptive[i](inputs[i])
      check(values.numel == 1 and indices.item == inputs[i].numel - 1, "adaptive max indices")
  close(f.adaptive_max_pool1d(inputs[0], 1), tensor([[[4.0]]], device:), "adaptive 1D function")

  (values, indices) = f.adaptive_max_pool2d(inputs[1], [null, 1], return_indices: true)
  close(values, tensor([[[[4.0], [8.0]]]], device:), "adaptive null axis")
  check(indices.to_a == [[[[3], [7]]]], "adaptive row indices")
  if device == "cpu":
    close(f.adaptive_max_pool3d(volume, 1), full([1, 1, 1, 1, 1], 8, device:), "adaptive 3D function")
  else:
    rejected(): f.adaptive_max_pool3d(volume, 1)
  odd = tensor([[[5.0, 1.0, 4.0, 2.0, 0.0]]], device:)

  (values, indices) = f.max_pool1d(odd, 2, return_indices: true)
  close(f.max_unpool1d(values, indices, 2, output_size: [5]), tensor([[[5.0, 0.0, 4.0, 0.0, 0.0]]], device:), "spatial output size")
  close(f.max_unpool1d(values, indices, 2, output_size: odd.shape),
    tensor([[[5.0, 0.0, 4.0, 0.0, 0.0]]], device:), "full output shape")

  (ceil_values, ceil_indices) = f.max_pool1d(odd, 2, ceil_mode: true, return_indices: true)
  close(ceil_values, tensor([[[5.0, 4.0, 0.0]]], device:), "pool ceil mode")
  check(f.max_unpool1d(ceil_values, ceil_indices, 2, output_size: [5]).shape == odd.shape, "ceil unpool output size")
  rejected(): f.max_unpool1d(values, indices, 2, output_size: [6])
  rejected(): f.max_unpool1d(values, indices, 2, output_size: [1, 5])
  rejected(): f.max_unpool1d(values, indices, 2, stride: 0)
  rejected(): f.max_unpool1d(values, indices.to(device:, dtype: "float32"), 2)
  rejected(): f.max_unpool1d(values, indices.reshape([2]), 2)
  rejected(): f.max_unpool1d(values, tensor([[[4, 0]]], dtype: "int64", device:), 2)
  close(f.max_unpool1d(values.snapshot, indices.snapshot, 2), tensor([[[5.0, 0.0, 4.0, 0.0]]], device:),
    "unpool accepts read-only inputs after invalid calls")

  lp = [LPPool1d(2, 2), LPPool2d(2, 2), LPPool3d(2, 2)]
  lp_inputs = [tensor([[[3.0, 4.0, 0.0, 0.0]]], device:), tensor([[[[3.0, 4.0], [0.0, 0.0]]]], device:),
    tensor([[[[[3.0, 4.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]], device:)]
  3.times |i|:
    input = lp_inputs[i].copy.requires_grad!(true)
    output = lp[i](input)
    check(output.flatten.to_a == (if i == 0 then [5.0, 0.0] else [5.0]), "LP pooling values")
    output.sum.backward!
    close(input.grad, input / 5, "LP pooling derivative and zero-window gradient")
    check(output.device.type == device and input.grad.device.type == device, "LP pooling backend")
  close(f.lp_pool1d(tensor([[[-3.0, -4.0]]], device:), 1, 2), tensor([[[-7.0]]], device:), "LP uses signed powers, not absolute inputs")
  close(f.lp_pool1d(tensor([[[3.0, 4.0, 5.0]]], device:), 2, 2, ceil_mode: true),
    tensor([[[5.0, 5.0 * (2.0 ** 0.5)]]], device:), "LP ceil boundary scaling")
  close(f.lp_pool2d(lp_inputs[1], 2, [2, 2]), full([1, 1, 1, 1], 5, device:), "LP 2D function")
  close(f.lp_pool3d(lp_inputs[2], 2, [2, 2, 2]), full([1, 1, 1, 1, 1], 5, device:), "LP 3D function")
  rejected(): f.lp_pool1d(inputs[0], 0, 2)
  print "PASS torch spatial " + device
  0

def main(): run("cpu")
