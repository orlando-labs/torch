package sputniktorch.onnx_operators_test

import sputniktorch
from sputniktorch.nn import Sequential, Identity, Linear, ConvTranspose2d, GroupNorm, LayerNorm
from sputniktorch.nn import GELU, SiLU, Mish, Softplus, Softshrink, Hardshrink, Hardswish
from sputniktorch.nn import ConstantPad2d, ReflectionPad2d, ReplicationPad2d
from sputniktorch.nn import AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from sputniktorch.nn import MultiheadAttention, GRU, LSTM, PReLU, Threshold
from sputniktorch.onnx import export_model, ONNXError
from sputniktorch.autograd import inference_mode, inference_enabled?, grad_enabled?
from sputniktorch.onnx_operator_model import OperatorModel
from sputniktorch.onnx_error_model import ErrorModel
from sputniktorch.onnx_tied_model import TiedModel
from sputniktorch.checks import check

export run, main

def _outputs(result):
  return [result] if sputniktorch.Tensor === result
  result.reduce([]) |items, entry|: items + _outputs(entry)

def _input(shape, batch, device, dtype):
  sample = sputniktorch.randn([batch, *shape], seed: batch + 11, device:)
  if dtype == "float32" then sample else sample.to(device:, dtype:)

def _verify(name, model, shape, device, dtype: "float32", kwargs: {}, opset_version: 18,
    dynamic_batch: true):
  model.eval!
  sample = _input(shape, 2, device, dtype)
  outputs = _outputs(model(sample, **kwargs))
  keyword_names = kwargs.keys.map: $it.str
  input_names = ["x", *keyword_names]
  output_names = Array.of(outputs.count): "out_#{$it}"
  axes = {}
  if dynamic_batch:
    axes["x"] = {0: "batch"}
    outputs.each_with_index |output, index|:
      axes[output_names[index]] = {0: "result_#{index}"} if output.rank > 0
  prefix = "build/onnx-#{device}-#{name}"
  program = export_model(model, sample, path: prefix + ".onnx",
    input_names:, output_names:, dynamic_axes: axes, kwargs:, opset_version:)
  check(program.byte_count > 0 and program.node_count > 0, "export #{name}")
  references = {}
  cases = if dynamic_batch then [1, 3] else [2]
  cases.each |batch|:
    input = _input(shape, batch, device, dtype)
    references["x.#{batch}"] = input
    kwargs.each |key, value|:
      references["#{key}.#{batch}"] = value
    _outputs(model(input, **kwargs)).each_with_index |output, index|:
      references["#{output_names[index]}.#{batch}"] = output
  sputniktorch.save(references, prefix + ".pt")

def _errors(device):
  sample = sputniktorch.randn([2, 4, 3, 3], seed: 0, device:)
  before = sample.to_a
  path = "build/onnx-#{device}-preserved.onnx"
  identity = Identity()
  identity.eval!
  valid = export_model(identity, sample, path:)
  [:item, :mutation, :backward, :grad, :state, :unsupported, :nested].each |operation|:
    model = ErrorModel(operation)
    model.eval!
    refused = false
    try:
      export_model(model, sample, path:)
    rescue ONNXError |error|:
      refused = true
    check(refused, "refuse #{operation}")
    check(grad_enabled?() and not inference_enabled?(), "restore failed trace scope")
    check(sample.to_a == before, "failed trace preserves input")
    check(fs.read_bytes(path).hex == valid.bytes.hex, "failed export preserves existing file")
  invalid_options = [
    {opset_version: 17}, {input_names: []}, {input_names: ["same"], output_names: ["same"]},
    {dynamic_axes: {"unknown": {0: "batch"}}}, {dynamic_axes: {"input_0": {4: "batch"}}}
  ]
  invalid_options.each |options|:
    refused = false
    try:
      export_model(identity, sample, **options)
    rescue ONNXError |error|:
      refused = true
    check(refused, "validate export options")
  nested = inference_mode():
    export_model(identity, sample)
    inference_enabled?()
  check(nested and grad_enabled?(), "restore enclosing inference scope")
  training = Linear(4, 4, device:)
  refused = false
  try:
    export_model(training, sample)
  rescue ONNXError |error|:
    refused = true
  check(refused and training.training?, "export requires explicit eval mode")
  tied = TiedModel(device)
  tied.eval!
  encoded = export_model(tied, sputniktorch.ones([2, 3], device:))
  check(encoded.initializer_count == 1, "tied parameters share an ONNX initializer")
  [[sample, sample], tied.weight].each |inputs|:
    refused = false
    try:
      export_model(tied, inputs)
    rescue ONNXError |error|:
      refused = true
    check(refused, "reject input identity collisions")

def run(device):
  _verify("identity", Identity(), [3], device)
  _verify("math", OperatorModel(:math), [3, 4], device)
  _verify("layout", OperatorModel(:layout), [3, 4], device, opset_version: 20)
  _verify("types", OperatorModel(:types), [3], device, dtype: "int64")
  _verify("task", OperatorModel(:task), [3], device, opset_version: 19)
  mask = sputniktorch.tensor([[true, true, false], [false, false, false], [true, true, true]], device:)
  _verify("attention", OperatorModel(:attention), [3, 4], device, kwargs: {mask:})
  _verify("causal", OperatorModel(:causal), [3, 4], device, kwargs: {mask:})
  normal = Sequential([GroupNorm(2, 4, device:), GELU(), SiLU(), Mish(), LayerNorm([6, 6], device:)])
  _verify("normalization", normal, [4, 6, 6], device)
  activations = Sequential([GELU(approximate: :tanh), Softplus(beta: 2.0, threshold: 1.0),
    Softshrink(), Hardshrink(), Hardswish(), PReLU(num_parameters: 4, device:), Threshold(0.1, 0.2)])
  _verify("activations", activations, [4, 6, 6], device)
  padding = Sequential([ConstantPad2d(1, 1.5), ReflectionPad2d(1), ReplicationPad2d(1)])
  _verify("padding", padding, [4, 6, 6], device)
  pooling = Sequential([MaxPool2d(2), AvgPool2d(2), AdaptiveAvgPool2d(1), AdaptiveMaxPool2d(1)])
  _verify("pooling", pooling, [4, 8, 8], device)
  _verify("transpose_conv", ConvTranspose2d(4, 2, 3, stride: 2, output_padding: 1, device:), [4, 6, 6], device)
  _verify("mha", MultiheadAttention(8, 2, device:), [3, 8], device, dynamic_batch: false)
  _verify("gru", GRU(4, 5, device:), [3, 4], device, dynamic_batch: false)
  _verify("lstm", LSTM(4, 5, device:), [3, 4], device, dynamic_batch: false)
  _errors(device)
  print "PASS ONNX operators + trace scope + atomic failure on #{device}"
  0

def main(): run("cpu")
