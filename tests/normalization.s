package torch.normalization_test

from torch import tensor, zeros, ones, full, no_grad
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d
from torch.nn import InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, LocalResponseNorm
from torch.nn import Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout, FeatureAlphaDropout
from torch.random.generator import Generator
from torch.autograd.context import with_trace
from torch.checks import check
import torch.nn.functional as f

export main, run

def close(x, expected, message, tolerance: 2e-4):
  check(x.shape == expected.shape and (x - expected).abs.sum.item < tolerance, message)

def rejected(&block):
  failed = false
  try:
    block()
  rescue ValueError:
    failed = true
  rescue ArgumentError:
    failed = true
  rescue Torch.Errors.Error:
    failed = true
  check(failed, "invalid normalization/regularization must raise")

def run(device):
  first = tensor([[1.0, 3.0], [3.0, 7.0]], device:)
  second = tensor([[5.0, 1.0], [7.0, 5.0]], device:)
  cumulative = BatchNorm1d(2, momentum: null, eps: 0, device:)
  rejected(): cumulative(first)
  cumulative.call!(first)
  close(cumulative.running_mean, tensor([2.0, 5.0], device:), "first cumulative mean")
  close(cumulative.running_var, tensor([2.0, 8.0], device:), "unbiased running variance")
  cumulative.call!(second)
  check(cumulative.num_batches_tracked.item == 2, "successful batch counter")
  close(cumulative.running_mean, tensor([4.0, 4.0], device:), "cumulative average")
  cumulative.eval!
  reference = (first - tensor([4.0, 4.0], device:)) / tensor([2.0, 8.0], device:).sqrt
  close(cumulative(first), reference, "eval uses cumulative statistics")
  saved = cumulative.state_dict
  restored = BatchNorm1d(2, momentum: null, eps: 0, device:)
  restored.load_state_dict!(saved).eval!
  close(restored(first), reference, "cumulative state round trip")
  restored.train!.call!(first)
  close(restored.running_mean, tensor([10.0 / 3, 13.0 / 3], device:), "resumed cumulative counter")
  restored.reset_running_stats!
  check(restored.num_batches_tracked.item == 0, "reset batch counter")
  close(restored.running_mean, zeros([2], device:), "reset running mean")
  close(restored.running_var, ones([2], device:), "reset running variance")
  no_grad():
    restored.weight.fill!(3)
    restored.bias.fill!(2)
  restored.reset_parameters!
  close(restored.weight, ones([2], device:), "reset affine scale")
  close(restored.bias, zeros([2], device:), "reset affine bias")
  rejected(): restored.call!(ones([1, 2], device:))
  rejected(): restored.call!(ones([2, 3], device:))
  check(restored.num_batches_tracked.item == 0, "failed batches do not advance counter")
  close(restored.running_mean, zeros([2], device:), "failed batch preserves buffers")
  fixed = BatchNorm1d(2, momentum: 0.25, device:)
  fixed.call!(first)
  close(fixed.running_mean, tensor([0.5, 1.25], device:), "fixed momentum regression")
  close(fixed.running_var, tensor([1.25, 2.75], device:), "fixed variance regression")
  stateless = BatchNorm1d(2, momentum: null, affine: false, track_running_stats: false, eps: 0, device:)
  close(stateless(first), tensor([[-1.0, -1.0], [1.0, 1.0]], device:), "stateless null momentum")
  check(stateless.parameters.empty? and stateless.named_buffers.empty?, "stateless registry")
  bn_layers = [BatchNorm1d(2, momentum: null, bias: false, device:),
    BatchNorm2d(2, momentum: null, bias: false, device:),
    BatchNorm3d(2, momentum: null, bias: false, device:)]
  bn_shapes = [[2, 2, 1], [2, 2, 1, 1], [2, 2, 1, 1, 1]]
  3.times |i|:
    input = first.reshape(bn_shapes[i]).copy.requires_grad!(true)
    output = bn_layers[i].call!(input)
    output.square.sum.backward!
    check(bn_layers[i].bias == null and bn_layers[i].parameters.count == 1, "bias-free BatchNorm")
    check(input.grad != null and bn_layers[i].weight.grad != null, "BatchNorm backward")
    check(output.device.type == device and input.grad.device.type == device, "BatchNorm backend")

  data = tensor([[[1.0, 3.0], [2.0, 6.0]], [[5.0, 7.0], [8.0, 12.0]]], device:)
  normalized = tensor([[[-1.0, 1.0], [-1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]], device:)
  shapes = [[2, 2, 2], [2, 2, 1, 2], [2, 2, 1, 1, 2]]
  instances = [InstanceNorm1d(2, eps: 0, device:), InstanceNorm2d(2, eps: 0, device:),
    InstanceNorm3d(2, eps: 0, device:)]
  tracked = [InstanceNorm1d(2, eps: 0, momentum: 1, affine: true, track_running_stats: true, device:),
    InstanceNorm2d(2, eps: 0, momentum: 1, affine: true, track_running_stats: true, device:),
    InstanceNorm3d(2, eps: 0, momentum: 1, affine: true, track_running_stats: true, device:)]
  3.times |i|:
    input = data.reshape(shapes[i]).copy.requires_grad!(true)
    expected = normalized.reshape(shapes[i])
    close(instances[i](input), expected, "per-instance statistics")
    instances[i].eval!
    close(instances[i](input), expected, "untracked eval uses input statistics")
    close(instances[i](input[0]), expected[0], "unbatched InstanceNorm")
    check(instances[i].parameters.empty? and instances[i].named_buffers.empty?, "default InstanceNorm registry")
    rejected(): tracked[i](input)
    output = tracked[i].call!(input)
    close(output, expected, "tracked training statistics")
    output.square.sum.backward!
    check(input.grad != null and tracked[i].weight.grad != null and tracked[i].bias.grad != null,
          "InstanceNorm input and affine backward")
    close(tracked[i].running_mean, tensor([4.0, 7.0], device:), "instance running mean averages samples")
    close(tracked[i].running_var, tensor([2.0, 8.0], device:), "instance unbiased running variance")
    check(tracked[i].num_batches_tracked.item == 0, "InstanceNorm counter stays zero")
    check(tracked[i].state_dict.count == 5, "InstanceNorm registered state")
    tracked[i].eval!
    evaluated = tracked[i](input).reshape([2, 2, 2])
    ref = ((data - tensor([4.0, 7.0], device:).reshape([1, 2, 1])) /
      tensor([2.0, 8.0], device:).sqrt.reshape([1, 2, 1]))
    close(evaluated, ref, "InstanceNorm eval running statistics")
    check(output.device.type == device and input.grad.device.type == device, "InstanceNorm backend")
    tracked[i].reset_parameters!
    close(tracked[i].running_var, ones([2], device:), "InstanceNorm resets inherited buffers")
  unchanged = InstanceNorm1d(2, momentum: null, track_running_stats: true, device:)
  unchanged.call!(data)
  close(unchanged.running_mean, zeros([2], device:), "InstanceNorm null momentum disables updates")
  close(unchanged.running_var, ones([2], device:), "InstanceNorm null variance unchanged")
  copy = InstanceNorm1d(2, affine: true, track_running_stats: true, eps: 0, device:)
  copy.load_state_dict!(tracked[0].state_dict).eval!
  close(copy(data), data, "InstanceNorm state round trip")
  rejected(): instances[0](ones([1, 2, 1], device:))
  rejected(): instances[0](ones([2], device:))
  rejected(): tracked[0](ones([1, 3, 2], device:))
  close(f.instance_norm(data, eps: 0), normalized, "functional InstanceNorm")
  mean = zeros([2], device:)
  variance = ones([2], device:)
  rejected(): f.instance_norm(data, running_mean: mean, running_var: variance)
  rejected(): f.instance_norm!(data, running_mean: mean, running_var: variance.snapshot)
  close(mean, zeros([2], device:), "read-only preflight before buffer updates")
  f.instance_norm!(data, running_mean: mean, running_var: variance, momentum: 1, eps: 0)
  close(mean, tensor([4.0, 7.0], device:), "functional InstanceNorm running buffers")
  rejected(): with_trace(77): f.instance_norm!(data, running_mean: mean, running_var: variance)
  rejected(): with_trace(77): fixed.call!(first)

  if device == "cpu":
    double = InstanceNorm1d(2, affine: true, bias: false, dtype: "float64", device:)
    output = double(data.to(device:, dtype: "float64"))
    check(output.dtype.to_str == "float64" and double.parameters.count == 1, "normalization dtype/bias options")
    check(BatchNorm1d(2, dtype: "float64", device:).running_var.dtype.to_str == "float64", "BatchNorm dtype")

  channels = ones([1, 3, 2], device:)
  close(LocalResponseNorm(3, alpha: 3, beta: 1)(channels),
    tensor([[[1.0 / 3, 1.0 / 3], [0.25, 0.25], [1.0 / 3, 1.0 / 3]]], device:), "odd LRN window")
  close(f.local_response_norm(channels, 2, alpha: 2, beta: 1),
    tensor([[[0.5, 0.5], [1.0 / 3, 1.0 / 3], [1.0 / 3, 1.0 / 3]]], device:), "even LRN window")
  image = channels.reshape([1, 3, 1, 2])
  close(f.local_response_norm(image, 3, alpha: 3, beta: 1).reshape(channels.shape),
    f.local_response_norm(channels, 3, alpha: 3, beta: 1), "LRN spatial layout")
  lrn_input = full([1, 1, 2], 1.3, device:, requires_grad: true)
  f.local_response_norm(lrn_input, 1, alpha: 0.2, beta: 0.5).sum.backward!
  base = 1 + 0.2 * 1.3 * 1.3
  derivative = base ** -0.5 - 0.2 * 1.3 * 1.3 * base ** -1.5
  close(lrn_input.grad, full([1, 1, 2], derivative, device:), "LRN closed-form derivative")
  empty = zeros([1, 0, 2], device:).snapshot
  check(f.local_response_norm(empty, 3)._identity == empty._identity, "empty LRN alias")
  rejected(): LocalResponseNorm(0)
  rejected(): f.local_response_norm(first, 3)

  cases = [[Dropout(p: 0.25, seed: 7), [2, 16, 3], :element],
    [Dropout1d(p: 0.25, seed: 7), [2, 16, 3], :channel],
    [Dropout2d(p: 0.25, seed: 7), [2, 16, 2, 3], :channel],
    [Dropout3d(p: 0.25, seed: 7), [2, 16, 2, 2, 3], :channel],
    [AlphaDropout(p: 0.25, seed: 7), [2, 16, 3], :alpha],
    [FeatureAlphaDropout(p: 0.25, seed: 7), [2, 16, 2, 3], :feature_alpha]]
  cases.each |entry|:
    layer = entry[0]
    input = ones(entry[1], device:, requires_grad: true)
    state = layer.rng_state
    rejected(): layer(input)
    output = layer.call!(input)
    layer.load_rng_state!(state)
    close(layer.call!(input), output, "dropout generator replay")
    output.sum.backward!
    expected_grad = output
    if entry[2] in [:alpha, :feature_alpha]:
      scale = ((1.7580993408473766 ** 2 * 0.25 + 1) * 0.75) ** -0.5
      expected_grad = output.gt(0).to(device:, dtype: "float32") * scale
    close(input.grad, expected_grad, "dropout derivative follows the sampled mask")
    close(input, ones(entry[1], device:), "bang dropout does not mutate input")
    if entry[2] in [:channel, :feature_alpha]:
      flat = output.reshape([32, -1])
      32.times |i|:
        check((flat[i] - flat[i][0].item).abs.sum.item < 1e-5, "whole-channel dropout mask")
    layer.eval!
    state = layer.rng_state.to_a
    check(layer(input)._identity == input._identity, "eval dropout is identity")
    check(layer.rng_state.to_a == state, "eval does not consume RNG")
    check(output.device.type == device and input.grad.device.type == device, "dropout backend")
  rng = Generator(seed: 19)
  keep = tensor([[true], [false]], device:)
  unbatched = ones([2, 3], device:, requires_grad: true)
  close(f.dropout1d(unbatched, mask: keep), tensor([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]], device:), "unbatched channel mask")
  close(f.dropout2d(ones([2, 2, 3], device:), mask: keep.reshape([2, 1, 1])).reshape([2, 6]),
    tensor([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device:), "LibTorch unbatched Dropout2d")
  check(f.dropout3d(ones([2, 2, 2, 3], device:), mask: keep.reshape([2, 1, 1, 1])).sum.item == 24,
        "unbatched Dropout3d")
  mask = tensor([true, false], device:)
  close(f.dropout(ones([2], device:), mask:), tensor([2.0, 0.0], device:), "pure element dropout")
  alpha = 1.7580993408473766
  scale = ((alpha * alpha * 0.5 + 1) * 0.5) ** -0.5
  reference = tensor([scale + alpha * scale * 0.5, -alpha * scale * 0.5], device:)
  close(f.alpha_dropout(ones([2], device:), training: true, mask:), reference, "alpha dropout saturation and rescale")
  close(f.feature_alpha_dropout(ones([1, 2, 3], device:), training: true, mask: mask.reshape([1, 2, 1])),
    reference.reshape([1, 2, 1]) * ones([1, 2, 3], device:), "pure feature alpha dropout")
  check(f.dropout!(first, rng).shape == first.shape, "functional dropout bang")
  check(f.dropout1d!(unbatched, rng).shape == unbatched.shape, "functional dropout1d bang")
  check(f.dropout2d!(image, rng).shape == image.shape, "functional dropout2d bang")
  check(f.dropout3d!(image.unsqueeze(0), rng).rank == 5, "functional dropout3d bang")
  check(f.alpha_dropout!(first, rng, training: true).shape == first.shape, "functional alpha bang")
  check(f.feature_alpha_dropout!(image, rng, training: true).shape == image.shape, "functional feature alpha bang")
  state = rng.state.to_a
  close(f.dropout!(first, rng, p: 1), zeros(first.shape, device:), "p=1 dropout")
  close(f.alpha_dropout!(first, rng, p: 1, training: true), zeros(first.shape, device:), "p=1 alpha dropout")
  check(f.dropout!(first, rng, p: 0)._identity == first._identity, "p=0 identity")
  check(f.feature_alpha_dropout!(first, rng)._identity == first._identity, "alpha default eval identity")
  check(f.dropout!(empty, rng)._identity == empty._identity, "empty dropout identity")
  rejected(): f.dropout1d!(first.reshape([4]), rng)
  rejected(): f.dropout!(tensor([1, 2], device:), rng)
  rejected(): with_trace(77): f.dropout!(first, rng)
  check(rng.state.to_a == state, "no-op and rejected dropout preserve RNG")
  rejected(): Dropout(p: -0.1)
  rejected(): f.dropout(first)
  rejected(): f.dropout(first, mask: tensor([[1.0, 0.0], [0.0, 1.0]], device:))
  rejected(): f.dropout1d(unbatched, mask: tensor([true, false], device:))
  print "PASS torch normalization/regularization " + device
  0

def main(): run("cpu")
