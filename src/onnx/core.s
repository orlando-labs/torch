package torch.onnx.core

from torch.errors import ONNXError

export TraceCore, open_trace, trace_token, finish_trace!, save_program

native class TraceCore from "torch.ONNXTraceCore" owned:
  def destroy!() from "amt_onnx_trace_free"

native def open_trace(inputs, state) from "amt_onnx_trace_open"
native def trace_token(trace) from "amt_onnx_trace_token"
native def finish_trace!(trace, outputs, options) from "amt_onnx_trace_finish"
native def save_program(bytes, path) from "amt_onnx_save" blocking
