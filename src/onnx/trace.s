package sputniktorch.onnx.trace

from sputniktorch.onnx.core import open_trace, trace_token, finish_trace!

export Trace

class Trace:
  def init(inputs, state):
    handles = inputs.map: $it._core
    entries = state.map |name, tensor|: [name, tensor._core]
    @core = open_trace(handles, entries)

  def token(): trace_token(@core)

  def finish!(outputs, options):
    handles = outputs.map: $it._core
    finish_trace!(@core, handles, options)

  def close!(): @core.destroy!
