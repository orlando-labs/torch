package torch.onnx.exporter

from torch.errors import ONNXError
from torch.tensor.value import Tensor
from torch.nn.module import Module
from torch.autograd.context import no_grad, with_trace
from torch.onnx.trace import Trace
from torch.onnx.program import Program

export export_model

def _outputs(value):
  return [value] if Tensor === value
  if Array === value or Tuple === value:
    value.reduce([]) |items, entry|: items + _outputs(entry)
  else:
    raise ONNXError("ONNX outputs must be tensors or arrays/tuples of tensors")

def _names(names, count, prefix):
  result = if names == null:
    Array.of(count): "#{prefix}_#{$it}"
  else:
    names.array
  raise ONNXError("#{prefix} name count does not match tensor count") if result.count != count
  result

def export_model(model, inputs, path: null, input_names: null, output_names: null,
    dynamic_axes: {}, kwargs: {}, opset_version: 18):
  raise ONNXError("ONNX export requires a Module") unless Module === model
  raise ONNXError("call model.eval! before ONNX export") if model.training?
  args = if Tensor === inputs then [inputs] else inputs.array
  raise ONNXError("ONNX export requires at least one input") if args.empty? and kwargs.empty?
  args.each |input|:
    raise ONNXError("positional ONNX inputs must be tensors") unless Tensor === input
  traced_inputs = args + kwargs.values.filter: Tensor === $it
  raise ONNXError("ONNX export requires tensor inputs") if traced_inputs.empty?
  input_names = _names(input_names, traced_inputs.count, "input")
  state = model.named_parameters.merge(model.named_buffers)
  trace = Trace(traced_inputs, state)
  try:
    result = no_grad():
      with_trace(trace.token): model(*args, **kwargs)
    outputs = _outputs(result)
    raise ONNXError("ONNX export requires tensor outputs") if outputs.empty?
    output_names = _names(output_names, outputs.count, "output")
    axes = dynamic_axes.map |name, dimensions|:
      [name, dimensions.map |axis, label|: [axis, label]]
    options = [opset_version, input_names, output_names, axes]
    encoded = trace.finish!(outputs, options)
    program = Program(*encoded, opset_version, input_names, output_names)
    program.save(path) if path != null
    program
  ensure:
    trace.close!
