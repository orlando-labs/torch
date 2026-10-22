package torch.attention_test

from torch import tensor, randn, zeros, ones
from torch.tensor.value import Tensor
from torch.nn import MultiheadAttention, Transformer, TransformerEncoderLayer, TransformerDecoderLayer, Sequential
from torch.nn.functional import scaled_dot_product_attention, scaled_dot_product_attention!, cat
from torch.nn.attention_functional import attention
from torch.random import Generator
from torch.autograd import no_grad, inference_mode
from torch.checks import check

export main, run

native def _reference(kind, inputs, parameters, mask, options) from "amt_attention_reference" blocking

def close(actual, expected, message):
  check(actual.shape == expected.shape, message + " shape")
  difference = (actual.to(device: "cpu") - expected.to(device: "cpu")).abs.sum.item
  check(difference < 8e-4, message + " values")

def rejected(&block):
  failed = false
  try:
    block()
  rescue ValueError:
    failed = true
  rescue TypeError:
    failed = true
  rescue Torch.Errors.Error:
    failed = true
  check(failed, "invalid attention call must raise")

def parameters(model):
  [model.query_projection.weight, model.key_projection.weight, model.value_projection.weight,
    model.output_projection.weight, model.query_projection.bias, model.key_projection.bias,
    model.value_projection.bias, model.output_projection.bias, model.bias_k, model.bias_v]

def compare_sdpa(device, dtype, kind):
  gqa = kind == "gqa"
  heads = if gqa then 4 else 2
  q = randn([2, heads, 3, 4], seed: 1, device:, dtype:, requires_grad: true)
  k = randn([1, 2, 5, 4], seed: 2, device:, dtype:, requires_grad: true)
  v = randn([2, 2, 5, 3], seed: 3, device:, dtype:, requires_grad: true)
  mask = null
  mask = tensor([[true, false, true, true, false], [true, true, false, true, true], [false, true, true, false, true]], device:) if kind == "bool"
  mask = randn([3, 5], seed: 4, device:, dtype:, requires_grad: true) if kind == "float"
  scale = if kind == "float" then 0.7 else null
  causal = kind == "causal"
  handle = if mask == null then null else mask._core
  expected = _reference("sdpa", [q._core, k._core, v._core], [], handle, [scale, causal, gqa])
  output = scaled_dot_product_attention(q, k, v, attn_mask: mask, is_causal: causal, scale:, enable_gqa: gqa)
  close(output, Tensor(expected[0]), "ATen SDPA " + kind)
  output.square.sum.backward!
  [q, k, v].each_with_index |x, index|:
    close(x.grad, Tensor(expected[2][index]), "ATen SDPA input gradient")
    check(x.grad.device.type == device, "SDPA gradient device")
  close(mask.grad, Tensor(expected[4]), "additive mask gradient") if kind == "float"
  check(output.device.type == device, "SDPA output device")

def compare_mha(device, dtype, bias, batch_first, extras):
  model = MultiheadAttention(4, 2, bias:, batch_first:, add_bias_kv: extras, add_zero_attn: extras, seed: 9, device:, dtype:)
  no_grad():
    model.query_projection.bias.fill!(0.04) if bias
    model.key_projection.bias.fill!(-0.02) if bias
    model.value_projection.bias.fill!(0.06) if bias
    model.output_projection.bias.fill!(-0.03) if bias
  qshape = if batch_first then [2, 3, 4] else [3, 2, 4]
  kvshape = if batch_first then [2, 5, 4] else [5, 2, 4]
  q = randn(qshape, seed: 5, device:, dtype:, requires_grad: true)
  k = randn(kvshape, seed: 6, device:, dtype:, requires_grad: true)
  v = randn(kvshape, seed: 7, device:, dtype:, requires_grad: true)
  mask = randn([3, 5], seed: 8, device:, dtype:, requires_grad: true)
  params = parameters(model)
  handles = params.map: if $it == null then null else $it._core
  expected = _reference("mha", [q._core, k._core, v._core], handles, mask._core, [batch_first, 2, extras])
  (output, weights) = model(q, key: k, value: v, attn_mask: mask, need_weights: true, average_attn_weights: false)
  close(output, Tensor(expected[0]), "LibTorch MHA output")
  close(weights, Tensor(expected[1]), "LibTorch MHA per-head weights")
  close(model(q, key: k, value: v, attn_mask: mask), output, "MHA SDPA and weight paths agree")
  averaged = model(q, key: k, value: v, attn_mask: mask, need_weights: true)[1]
  close(averaged, weights.mean(dim: 1), "MHA average weights")
  (output.square.sum + weights.square.sum).backward!
  [q, k, v].each_with_index |x, index|:
    close(x.grad, Tensor(expected[2][index]), "LibTorch MHA input gradient")
  params.each_with_index |parameter, index|:
    if parameter != null:
      close(parameter.grad, Tensor(expected[3][index]), "LibTorch MHA parameter gradient")
      check(parameter.leaf? and parameter.grad.device.type == device, "MHA parameter device")
  close(mask.grad, Tensor(expected[4]), "LibTorch MHA mask gradient")
  check(output.device.type == device and weights.device.type == device, "MHA output devices")

def run(device):
  ["plain", "bool", "float", "causal", "gqa"].each: compare_sdpa(device, "float32", $it)
  compare_sdpa(device, "float64", "float") if device == "cpu"
  compare_mha(device, "float32", true, true, true)
  compare_mha(device, "float32", false, false, false)
  compare_mha(device, "float64", true, false, false) if device == "cpu"

  x = randn([2, 3, 4], seed: 1, device:)
  model = MultiheadAttention(4, 2, device:)
  blocked = tensor([[false, true, false], [false, false, true], [true, false, false]], device:)
  padding = tensor([[false, true, false], [true, false, false]], device:)
  allowed = blocked.eq(false) * padding.eq(false).reshape([2, 1, 1, 3])
  close(model(x, attn_mask: blocked, key_padding_mask: padding), model(x, mask: allowed), "Torch blocked masks versus legacy allowed mask")
  bias = randn([3, 3], seed: 8, device:)
  (combined, combined_weights) = model(x, attn_mask: bias, key_padding_mask: padding, need_weights: true, average_attn_weights: false)
  2.times |i|:
    single = model(x[i], attn_mask: bias, key_padding_mask: padding[i], need_weights: true, average_attn_weights: false)
    close(combined[i], single[0], "mixed bool/additive mask batch isolation")
    close(combined_weights[i], single[1], "unbatched per-head weights")
  per_head = cat([blocked.unsqueeze(0), blocked.eq(false).unsqueeze(0), blocked.unsqueeze(0), blocked.unsqueeze(0)])
  weighted = model(x, attn_mask: per_head, need_weights: true, average_attn_weights: false)
  check(weighted[1].shape == [2, 2, 3, 3], "3D per-head mask layout")
  close(weighted[1][0][0] * blocked, zeros([3, 3], device:), "per-head blocked weights")
  close(weighted[1][0][1] * blocked.eq(false), zeros([3, 3], device:), "different per-head blocked weights")
  handles = parameters(model).map: if $it == null then null else $it._core
  expected = _reference("mha", [x._core, x._core, x._core], handles, blocked._core, [true, 2, false])
  close(model(x, attn_mask: blocked), Tensor(expected[0]), "LibTorch boolean blocked mask")
  extras = MultiheadAttention(4, 2, add_bias_kv: true, add_zero_attn: true, device:)
  extra_weights = extras(x, mask: zeros([3, 3], dtype: "bool", device:), causal: true,
    need_weights: true, average_attn_weights: false)[1]
  close(extra_weights.narrow(-1, 0, 3), zeros([2, 2, 3, 3], device:), "extra slots preserve original blocked columns")
  close(extra_weights.narrow(-1, 3, 2).sum(dim: -1), ones([2, 2, 3], device:), "extra slots stay visible under causal masks")

  separate = MultiheadAttention(4, 2, kdim: 3, vdim: 5, bias: false, device:)
  reference = MultiheadAttention(4, 2, kdim: 5, vdim: 5, bias: false, device:)
  no_grad():
    reference.query_projection.weight.copy_from!(separate.query_projection.weight)
    reference.key_projection.weight.copy_from!(cat([separate.key_projection.weight, zeros([4, 2], device:)], dim: 1))
    reference.value_projection.weight.copy_from!(separate.value_projection.weight)
    reference.output_projection.weight.copy_from!(separate.output_projection.weight)
  key = randn([2, 5, 3], seed: 4, device:, requires_grad: true)
  value = randn([2, 5, 5], seed: 5, device:, requires_grad: true)
  padded_key = cat([key, zeros([2, 5, 2], device:)], dim: -1)
  close(separate(x, key:, value:), reference(x, key: padded_key, value:), "separate key/value dimensions")
  handles = parameters(reference).map: if $it == null then null else $it._core
  expected = _reference("mha", [x._core, padded_key._core, value._core], handles, null, [true, 2, false])
  separate_result = separate(x, key:, value:, need_weights: true, average_attn_weights: false)
  close(separate_result[0], Tensor(expected[0]), "separate dimensions LibTorch output")
  close(separate_result[1], Tensor(expected[1]), "separate dimensions LibTorch weights")
  (separate_result[0].square.sum + separate_result[1].square.sum).backward!
  check(key.grad.shape == key.shape and value.grad.shape == value.shape, "separate key/value gradients")
  close(key.grad, Tensor(expected[2][1]).narrow(-1, 0, 3), "separate dimensions LibTorch key gradient")
  close(value.grad, Tensor(expected[2][2]), "separate dimensions LibTorch value gradient")
  close(separate.key_projection.weight.grad, Tensor(expected[3][1]).narrow(-1, 0, 3), "separate dimensions LibTorch key parameter gradient")

  q = randn([1, 2, 3, 4], seed: 2, device:, requires_grad: true)
  v = randn([1, 2, 3, 4], seed: 3, device:, requires_grad: true)
  closed = zeros([3, 3], dtype: "bool", device:)
  result = scaled_dot_product_attention(q, q, v, mask: closed)
  close(result, zeros(q.shape, device:), "fully masked fused SDPA rows")
  result.sum.backward!
  close(q.grad, zeros(q.shape, device:), "fully masked fused query gradient")
  close(v.grad, zeros(v.shape, device:), "fully masked fused value gradient")
  q.zero_grad!
  v.zero_grad!
  composed = attention(q, q, v, mask: closed, need_weights: true)
  close(composed[0], result, "fully masked composed output")
  composed[0].sum.backward!
  close(q.grad, zeros(q.shape, device:), "fully masked composed query gradient")
  negative_infinity = zeros([3, 3], device:).log
  close(scaled_dot_product_attention(q, q, v, mask: negative_infinity), result, "fully masked additive SDPA rows")
  close(attention(q, q, v, mask: negative_infinity, need_weights: true)[0], result, "fully masked additive composed rows")
  empty_rows = scaled_dot_product_attention(zeros([1, 2, 0, 4], device:), q, v)
  check(empty_rows.shape == [1, 2, 0, 4], "empty query sequence")

  noisy = MultiheadAttention(4, 2, dropout: 0.5, seed: 20, device:)
  before = noisy.rng_state
  rejected(): noisy(x)
  close(noisy.rng_state, before, "non-bang MHA preserves RNG")
  first = noisy.call!(x, need_weights: true)
  check((noisy.rng_state - before).abs.sum.item > 0, "MHA dropout consumes RNG")
  noisy.load_rng_state!(before)
  replay = noisy.call!(x, need_weights: true)
  close(replay[0], first[0], "MHA dropout replay output")
  close(replay[1], first[1], "MHA dropout replay weights")
  before_invalid = noisy.rng_state
  rejected(): noisy.call!(x, key_padding_mask: zeros([1, 3], dtype: "bool", device:))
  close(noisy.rng_state, before_invalid, "invalid MHA leaves RNG unchanged")
  rejected(): Sequential([noisy])(x)
  close(noisy.rng_state, before_invalid, "container stochastic preflight")
  noisy.eval!
  close(noisy(x), MultiheadAttention(4, 2, seed: 20, device:)(x), "eval disables attention dropout")
  close(noisy.rng_state, before_invalid, "eval leaves RNG unchanged")
  dropped = MultiheadAttention(4, 2, dropout: 1, bias: false, device:)
  close(dropped(x), zeros(x.shape, device:), "attention dropout p1")
  rng = Generator(seed: 22)
  rng_before = rng.state
  stochastic = scaled_dot_product_attention!(q, q, v, rng, dropout_p: 0.5)
  rng.load_state!(rng_before)
  close(scaled_dot_product_attention!(q, q, v, rng, dropout_p: 0.5), stochastic, "functional SDPA dropout replay")
  mask = ones([1, 2, 3, 3], dtype: "bool", device:)
  close(scaled_dot_product_attention(q, q, v, dropout_p: 0.5, dropout_mask: mask), scaled_dot_product_attention(q, q, v) * 2, "explicit pure attention dropout mask")
  rejected(): scaled_dot_product_attention(q, q, v, dropout_p: 0.5)

  clean = Transformer(4, 2, num_encoder_layers: 2, num_decoder_layers: 2, dim_feedforward: 6,
    dropout: 0, bias: false, activation: :relu, layer_norm_eps: 1e-4, device:)
  time_first = Transformer(4, 2, num_encoder_layers: 2, num_decoder_layers: 2, dim_feedforward: 6,
    dropout: 0, bias: false, activation: :relu, layer_norm_eps: 1e-4, batch_first: false, device:)
  time_first.load_state_dict!(clean.state_dict)
  result = clean(x, x, source_key_padding_mask: padding, target_key_padding_mask: padding, memory_key_padding_mask: padding)
  close(time_first(x.transpose(0, 1).snapshot, x.transpose(0, 1).snapshot,
    source_key_padding_mask: padding, target_key_padding_mask: padding, memory_key_padding_mask: padding).transpose(0, 1), result, "Transformer time-first and padding masks")
  close(clean(x[0], x[0]), clean(x[0].unsqueeze(0), x[0].unsqueeze(0)).squeeze(dim: 0), "unbatched Transformer")
  result.square.sum.backward!
  check(clean.parameters.all?: $it.grad != null, "Transformer all parameter gradients")
  encoder = TransformerEncoderLayer(4, 2, dim_feedforward: 6, dropout: 0.5, device:)
  decoder = TransformerDecoderLayer(4, 2, dim_feedforward: 6, dropout: 0.5, norm_first: true, device:)
  rejected(): encoder(x)
  encoder.call!(x).square.sum.backward!
  decoder.call!(x, x).square.sum.backward!
  check(encoder.parameters.all?: $it.grad != null, "encoder attention dropout bang propagation")
  check(decoder.parameters.all?: $it.grad != null, "decoder attention dropout bang propagation")
  if device == "cpu":
    double_layer = TransformerEncoderLayer(4, 2, dim_feedforward: 6, dropout: 0, dtype: "float64")
    check(double_layer(x.to(device:, dtype: "float64")).dtype.to_str == "float64", "Transformer dtype forwarding")
  detached = no_grad(): model(x)
  check(not detached.requires_grad?, "attention no-grad scope")
  inferred = inference_mode(): model(x)
  check(not inferred.requires_grad?, "attention inference scope")
  rejected(): MultiheadAttention(4, 3, device:)
  rejected(): MultiheadAttention(4, 2, kdim: 0, device:)
  rejected(): MultiheadAttention(4, 2, dropout: -0.1, device:)
  rejected(): model(x, attn_mask: blocked, mask: blocked)
  rejected(): model(x, causal: true, is_causal: false)
  rejected(): model(x, attn_mask: zeros([2, 3, 3], dtype: "bool", device:))
  rejected(): model(x, mask: zeros([3, 3, 3], dtype: "bool", device:))
  rejected(): model(x, key: zeros([2, 2, 4], device:), value: x)
  rejected(): model(x, attn_mask: zeros([3, 3], dtype: "int32", device:))
  rejected(): scaled_dot_product_attention(q, q, v, mask: zeros([4, 3], dtype: "bool", device:))
  rejected(): scaled_dot_product_attention(q, q, v, enable_gqa: true, scale: "bad")
  rejected(): scaled_dot_product_attention(q, zeros([1, 3, 3, 4], device:), v, enable_gqa: true)
  if device == "mps":
    rejected(): model(x, attn_mask: zeros([3, 3], dtype: "bool"))
  print "PASS torch attention " + device
  0

def main():
  parser = ArgParser(name: "Attention checks")
  parser.arg("--device", choices: ["cpu", "mps"], default: "cpu")
  run(parser.parse_or_raise["device"])
