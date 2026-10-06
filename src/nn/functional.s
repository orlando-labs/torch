package sputniktorch.nn.functional

from sputniktorch.autograd.context import execution_mode
from sputniktorch.tensor.value import Tensor
from sputniktorch.tensor.factories import arange

export kernel, kernel_cores, spatial, linear, relu, sigmoid, tanh, gelu, silu, mish, leaky_relu, elu, selu
export softplus, softsign, hardtanh, hardsigmoid, hardswish, log_sigmoid, softshrink, hardshrink, tanhshrink
export softmax, log_softmax, embedding, conv1d, conv2d, conv3d, layer_norm, rms_norm, group_norm
export cat, stack, where, pad, mse_loss, l1_loss, cross_entropy, nll_loss, binary_cross_entropy
export binary_cross_entropy_with_logits, smooth_l1_loss, huber_loss, kl_div, reduction_code
export batch_norm, batch_norm!
export scaled_dot_product_attention

native def _kernel(op as Str, inputs, options, mode as Int) from "amt_nn"

native def _batch_norm!(x, weight, bias, mean, variance, training as Bool,
  momentum, eps, mode as Int) from "amt_batch_norm"

def kernel(op, inputs, options = []):
  handles = inputs.map: if $it == null then null else $it._core
  kernel_cores(op, handles, options)

def kernel_cores(op, handles, options): Tensor(_kernel(op, handles, options, execution_mode()))

def _core_or_null(value): if null === value then null else value._core

def spatial(value, dimensions):
  result = if Int === value then Array.filled(dimensions, value) else value.array
  raise ValueError("expected #{dimensions} spatial values") if result.count != dimensions
  result

def linear(input, weight, bias: null):
  output = input.matmul(weight.transpose(-1, -2))
  if bias == null then output else output + bias

def relu(x): x.relu
def sigmoid(x): x.sigmoid
def tanh(x): x.tanh
def softmax(x, dim: -1): x.softmax(dim:)
def log_softmax(x, dim: -1): x.log_softmax(dim:)
def gelu(x, approximate: :none): kernel("gelu", [x], [approximate.str])
def silu(x): kernel("silu", [x])
def mish(x): kernel("mish", [x])
def leaky_relu(x, negative_slope: 0.01): kernel("leaky_relu", [x], [negative_slope])
def elu(x, alpha: 1.0): kernel("elu", [x], [alpha])
def selu(x): kernel("selu", [x])
def softplus(x, beta: 1.0, threshold: 20.0): kernel("softplus", [x], [beta, threshold])
def softsign(x): x / (x.abs + 1)
def tanhshrink(x): x - x.tanh
def hardtanh(x, min_value: -1.0, max_value: 1.0): kernel("hardtanh", [x], [min_value, max_value])
def hardsigmoid(x): kernel("hardsigmoid", [x])
def hardswish(x): kernel("hardswish", [x])
def log_sigmoid(x): kernel("log_sigmoid", [x])
def softshrink(x, threshold: 0.5): kernel("softshrink", [x], [threshold])
def hardshrink(x, threshold: 0.5): kernel("hardshrink", [x], [threshold])

def embedding(indices, weight, padding_idx: -1): kernel("embedding", [indices, weight], [padding_idx])

def conv1d(x, weight, bias: null, stride: 1, padding: 0, dilation: 1, groups: 1):
  kernel("conv1d", [x, weight, bias],
    [spatial(stride, 1), spatial(padding, 1), spatial(dilation, 1), groups])

def conv2d(x, weight, bias: null, stride: 1, padding: 0, dilation: 1, groups: 1):
  kernel("conv2d", [x, weight, bias],
    [spatial(stride, 2), spatial(padding, 2), spatial(dilation, 2), groups])

def conv3d(x, weight, bias: null, stride: 1, padding: 0, dilation: 1, groups: 1):
  kernel("conv3d", [x, weight, bias],
    [spatial(stride, 3), spatial(padding, 3), spatial(dilation, 3), groups])

def layer_norm(x, normalized_shape, weight: null, bias: null, eps: 1e-5):
  kernel("layer_norm", [x, weight, bias], [normalized_shape.array, eps])

def rms_norm(x, normalized_shape, weight: null, eps: 1e-5):
  dims = normalized_shape.array.count
  variance = x.square
  dims.times |i|:
    variance = variance.mean(dim: -i - 1, keepdim: true)
  output = x * (variance + eps).rsqrt
  if weight == null then output else output * weight

def group_norm(x, groups, weight: null, bias: null, eps: 1e-5):
  kernel("group_norm", [x, weight, bias], [groups, eps])

def batch_norm(x, running_mean, running_var, weight: null, bias: null,
    training: false, momentum: 0.1, eps: 1e-5):
  kernel("batch_norm", [x, weight, bias, running_mean, running_var], [training, momentum, eps])

def batch_norm!(x, running_mean, running_var, weight: null, bias: null,
    momentum: 0.1, eps: 1e-5):
  Tensor(_batch_norm!(x._core, _core_or_null(weight), _core_or_null(bias),
    _core_or_null(running_mean), _core_or_null(running_var), true, momentum, eps, execution_mode()))

def cat(tensors, dim: 0): kernel("cat", tensors, [dim])
def stack(tensors, dim: 0): kernel("stack", tensors, [dim])
def where(condition, yes, no): kernel("where", [condition, yes, no])
def pad(x, padding, mode: :constant, value: 0.0): kernel("pad", [x], [padding.array, mode.str, value])

# Composition stays visible in Sputnik. True in a boolean mask means allowed.
# Fully masked rows produce zero values and zero query/key gradients.
def scaled_dot_product_attention(query, key, value, mask: null, causal: false):
  scale = query.shape[-1].float ** 0.5
  scores = query.matmul(key.transpose(-1, -2)) / scale
  allowed = mask
  if causal:
    rows = arange(query.shape[-2], device: query.device).unsqueeze(-1)
    columns = arange(key.shape[-2], device: key.device).unsqueeze(0)
    triangular = rows.ge(columns)
    allowed = if mask == null then triangular else triangular * mask
  probabilities = if allowed == null:
    scores.softmax(dim: -1)
  else:
    kernel("masked_softmax", [scores, allowed], [-1])
  probabilities.matmul(value)

def reduction_code(reduction):
  case! reduction.str:
    when "none": 0
    when "mean": 1
    when "sum": 2
    else: raise ValueError("reduction must be :none, :mean or :sum")

def _reduce(loss, reduction):
  case! reduction_code(reduction):
    when 0: loss
    when 1: loss.mean()
    when 2: loss.sum()

def mse_loss(input, target, reduction: :mean): _reduce((input - target).square, reduction)
def l1_loss(input, target, reduction: :mean): _reduce((input - target).abs, reduction)

def cross_entropy(input, target, weight: null, reduction: :mean,
    ignore_index: -100, label_smoothing: 0.0):
  kernel("cross_entropy", [input, target, weight],
    [reduction_code(reduction), ignore_index, label_smoothing])

def nll_loss(input, target, weight: null, reduction: :mean, ignore_index: -100):
  kernel("nll_loss", [input, target, weight], [reduction_code(reduction), ignore_index])

def binary_cross_entropy(input, target, weight: null, reduction: :mean):
  kernel("bce", [input, target, weight], [reduction_code(reduction)])

def binary_cross_entropy_with_logits(input, target, weight: null, pos_weight: null, reduction: :mean):
  kernel("bce_logits", [input, target, weight, pos_weight], [reduction_code(reduction)])

def smooth_l1_loss(input, target, beta: 1.0, reduction: :mean):
  kernel("smooth_l1", [input, target], [reduction_code(reduction), beta])

def huber_loss(input, target, delta: 1.0, reduction: :mean):
  kernel("huber", [input, target], [reduction_code(reduction), delta])

def kl_div(input, target, log_target: false, reduction: :mean):
  kernel("kl_div", [input, target], [reduction_code(reduction), log_target])
