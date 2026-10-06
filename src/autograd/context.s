package sputniktorch.autograd.context

import task
from sputniktorch.errors import ONNXError
export grad_enabled?, inference_enabled?, no_grad, with_grad, inference_mode, execution_mode
export with_trace

# Bound to the logical Sputnik task, including across blocking FFI migration.
MODE = task.local(inherit: true)

def execution_mode(): MODE.get(default: 0)

def grad_enabled?(): execution_mode() % 4 == 0
def inference_enabled?(): execution_mode() % 4 == 2

def _with_mode(mode, &block):
  trace = execution_mode() // 4
  MODE.with(mode + 4 * trace, &block)

def no_grad(&block): _with_mode(1, &block)
def with_grad(&block): _with_mode(0, &block)
def inference_mode(&block): _with_mode(2, &block)

def with_trace(token, &block):
  mode = execution_mode()
  raise ONNXError("nested ONNX tracing is unsupported") if mode // 4 != 0
  MODE.with(mode % 4 + 4 * token, &block)
