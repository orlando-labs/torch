package torch.nn.functional

from torch.autograd.context import execution_mode, no_grad
from torch.tensor.value import Tensor
from torch.tensor.factories import full_like
from torch.nn.dropout_functional import dropout, dropout!, dropout1d, dropout1d!
from torch.nn.dropout_functional import dropout2d, dropout2d!, dropout3d, dropout3d!
from torch.nn.dropout_functional import alpha_dropout, alpha_dropout!, feature_alpha_dropout, feature_alpha_dropout!
from torch.nn.spatial_core import spatial
from torch.nn.interpolation import interpolate
from torch.nn.pooling_functional import max_pool1d, max_pool2d, max_pool3d
from torch.nn.pooling_functional import adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d
from torch.nn.pooling_functional import max_unpool1d, max_unpool2d, max_unpool3d, lp_pool1d, lp_pool2d, lp_pool3d
from torch.nn.embedding_functional import embedding, embedding!, embedding_bag, embedding_bag!
from torch.nn.embedding_functional import embedding_renorm!, embedding_renormed
from torch.nn.attention_functional import scaled_dot_product_attention, scaled_dot_product_attention!

export kernel, kernel_cores, spatial, linear, relu, sigmoid, tanh, gelu, silu, mish, leaky_relu, elu, selu
export softplus, softsign, hardtanh, hardsigmoid, hardswish, log_sigmoid, softshrink, hardshrink, tanhshrink
export softmax, log_softmax, embedding, conv1d, conv2d, conv3d, layer_norm, rms_norm, group_norm
export cat, stack, where, pad, mse_loss, l1_loss, cross_entropy, nll_loss, binary_cross_entropy
export binary_cross_entropy_with_logits, smooth_l1_loss, huber_loss, kl_div, reduction_code
export batch_norm, batch_norm!, instance_norm, instance_norm!, local_response_norm
export dropout, dropout!, dropout1d, dropout1d!, dropout2d, dropout2d!, dropout3d, dropout3d!
export alpha_dropout, alpha_dropout!, feature_alpha_dropout, feature_alpha_dropout!
export scaled_dot_product_attention, scaled_dot_product_attention!
export interpolate, max_pool1d, max_pool2d, max_pool3d
export adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d
export max_unpool1d, max_unpool2d, max_unpool3d, lp_pool1d, lp_pool2d, lp_pool3d
export poisson_nll_loss, gaussian_nll_loss, soft_margin_loss, multilabel_soft_margin_loss
export hinge_embedding_loss, margin_ranking_loss, cosine_embedding_loss
export multi_margin_loss, multilabel_margin_loss, triplet_margin_loss, triplet_margin_with_distance_loss
export ctc_loss, pairwise_distance, cosine_similarity
export embedding!, embedding_bag, embedding_bag!, embedding_renorm!, embedding_renormed

native def _kernel(op as Str, inputs, options, mode as Int) from "amt_nn"

native def _batch_norm!(x, weight, bias, mean, variance, training as Bool,
  momentum, eps, mode as Int) from "amt_batch_norm"

native def _instance_norm!(x, weight, bias, mean, variance, use_input_stats as Bool,
  momentum, eps, mode as Int) from "amt_instance_norm"

native def _ctc_loss(log_probs, targets, input_lengths, target_lengths,
  blank as Int, reduction as Int, zero_infinity as Bool, mode as Int) from "amt_ctc_loss" blocking

def kernel(op, inputs, options = []):
  handles = inputs.map: if $it == null then null else $it._core
  kernel_cores(op, handles, options)

def kernel_cores(op, handles, options): Tensor(_kernel(op, handles, options, execution_mode()))

def _core_or_null(value): if null === value then null else value._core

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

def instance_norm(x, running_mean: null, running_var: null, weight: null, bias: null,
    use_input_stats: true, momentum: 0.1, eps: 1e-5):
  kernel("instance_norm", [x, weight, bias, running_mean, running_var], [use_input_stats, momentum, eps])

def instance_norm!(x, running_mean: null, running_var: null, weight: null, bias: null,
    momentum: 0.1, eps: 1e-5):
  Tensor(_instance_norm!(x._core, _core_or_null(weight), _core_or_null(bias),
    _core_or_null(running_mean), _core_or_null(running_var), true, momentum, eps, execution_mode()))

def local_response_norm(x, size, alpha: 1e-4, beta: 0.75, k: 1.0):
  raise ValueError("size must be a positive Int") unless Int === size and size > 0
  raise ValueError("LocalResponseNorm requires rank >= 3") if x.rank < 3
  return x if x.numel == 0
  channels = x.shape[1]
  squares = x.square.reshape([x.shape[0], channels, -1]).unsqueeze(1)
  squares = pad(squares, [0, 0, size // 2, (size - 1) // 2])
  average = kernel("avg_pool2d", [squares], [[size, 1], [1, 1], [0, 0], false, true])
  x / (average.reshape(x.shape) * alpha + k).pow(beta)

def cat(tensors, dim: 0): kernel("cat", tensors, [dim])
def stack(tensors, dim: 0): kernel("stack", tensors, [dim])
def where(condition, yes, no): kernel("where", [condition, yes, no])
def pad(x, padding, mode: :constant, value: 0.0): kernel("pad", [x], [padding.array, mode.str, value])

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
  if reduction.str == "batchmean":
    loss = kernel("kl_div", [input, target], [2, log_target])
    return if input.rank == 0 then loss else loss / input.shape[0]
  kernel("kl_div", [input, target], [reduction_code(reduction), log_target])

def poisson_nll_loss(input, target, log_input: true, full: false, eps: 1e-8, reduction: :mean):
  kernel("poisson_nll", [input, target], [log_input, full, eps, reduction_code(reduction)])

def gaussian_nll_loss(input, target, var, full: false, eps: 1e-6, reduction: :mean):
  reduction_code(reduction)
  variance = if Tensor === var then var else full_like(input, var)
  raise ValueError("variance must be nonnegative") if variance.lt(0).sum.item > 0
  if variance.shape != input.shape:
    if input.rank > 0 and variance.shape == input.shape.first(input.rank - 1):
      variance = variance.unsqueeze(-1)
    elif input.rank == variance.rank:
      mismatches = 0
      singleton = true
      input.rank.times |i|:
        if variance.shape[i] != input.shape[i]:
          mismatches += 1
          singleton = false unless variance.shape[i] == 1
      raise ValueError("variance has the wrong shape") unless singleton and mismatches == 1
    else:
      raise ValueError("variance has the wrong shape")
  # Stability clamping is intentionally ignored by the variance derivative.
  safe = variance.copy
  no_grad(): safe.copy_from!(kernel("clamp_min", [variance], [eps]))
  loss = (safe.log + (input - target).square / safe) * 0.5
  loss = loss + 0.5 * Math.log(2 * Math.PI) if full
  _reduce(loss, reduction)

def soft_margin_loss(input, target, reduction: :mean):
  kernel("soft_margin", [input, target], [reduction_code(reduction)])

def multilabel_soft_margin_loss(input, target, weight: null, reduction: :mean):
  raise ValueError("multilabel soft-margin requires a class axis") if input.rank == 0
  loss = (target * log_sigmoid(input) + (target * -1 + 1) * log_sigmoid(input.neg)).neg
  loss = loss * weight if weight != null
  _reduce(loss.sum(dim: -1) / input.shape[-1], reduction)

def hinge_embedding_loss(input, target, margin: 1.0, reduction: :mean):
  kernel("hinge_embedding", [input, target], [margin, reduction_code(reduction)])

def margin_ranking_loss(input1, input2, target, margin: 0.0, reduction: :mean):
  kernel("margin_ranking", [input1, input2, target], [margin, reduction_code(reduction)])

def cosine_embedding_loss(input1, input2, target, margin: 0.0, reduction: :mean):
  kernel("cosine_embedding", [input1, input2, target], [margin, reduction_code(reduction)])

def multi_margin_loss(input, target, p: 1, margin: 1.0, weight: null, reduction: :mean):
  raise ValueError("multi-margin p must be 1 or 2") unless p in [1, 2]
  kernel("multi_margin", [input, target, weight], [p, margin, reduction_code(reduction)])

def multilabel_margin_loss(input, target, reduction: :mean):
  kernel("multilabel_margin", [input, target], [reduction_code(reduction)])

def pairwise_distance(input1, input2, p: 2.0, eps: 1e-6, keepdim: false):
  kernel("pairwise_distance", [input1, input2], [p, eps, keepdim])

def cosine_similarity(input1, input2, dim: 1, eps: 1e-8):
  kernel("cosine_similarity", [input1, input2], [dim, eps])

def triplet_margin_loss(anchor, positive, negative, margin: 1.0, p: 2.0,
    eps: 1e-6, swap: false, reduction: :mean):
  raise ValueError("triplet margin must be positive") unless margin > 0
  kernel("triplet_margin", [anchor, positive, negative], [margin, p, eps, swap, reduction_code(reduction)])

def triplet_margin_with_distance_loss(anchor, positive, negative,
    distance_function: null, margin: 1.0, swap: false, reduction: :mean):
  reduction_code(reduction)
  raise ValueError("triplet margin must be positive") unless margin > 0
  unless anchor.rank == positive.rank and positive.rank == negative.rank:
    raise ValueError("triplet inputs must have the same rank")
  distance = if distance_function == null then &pairwise_distance else distance_function
  pos = distance(anchor, positive)
  neg = distance(anchor, negative)
  neg = kernel("minimum", [neg, distance(positive, negative)]) if swap
  loss = kernel("clamp_min", [pos - neg + margin], [0])
  _reduce(loss, reduction)

def _ctc_lengths(value): if Tensor === value then value._core else value.array

def ctc_loss(log_probs, targets, input_lengths, target_lengths,
    blank: 0, reduction: :mean, zero_infinity: false):
  Tensor(_ctc_loss(log_probs._core, targets._core, _ctc_lengths(input_lengths),
    _ctc_lengths(target_lengths), blank, reduction_code(reduction), zero_infinity, execution_mode()))
