package sputniktorch.tensor.value

from sputniktorch.tensor.core import meta, unary, binary, transfer, mutate!, backward! as _backward!
from sputniktorch.device import Device, device_text
from sputniktorch.dtype import DType

export Tensor

def _operand(value):
  if Tensor === value:
    value._core
  else:
    value

class Tensor:
  def init(@core)
  attr _core from @core
  def _identity(): @core.identity
  def init_copy(source): @core = unary(source._core, "copy", null)
  def shape(): @core.shape
  def strides(): @core.strides
  def rank(): @core.rank
  def numel(): @core.numel
  def dtype(): DType(@core.dtype)
  def device(): Device(@core.device)
  def requires_grad?(): @core.requires_grad?
  def leaf?(): @core.leaf?
  def readonly?(): @core.readonly?
  def contiguous?(): @core.contiguous?
  def item(): meta(@core, "item")
  def to_a(): meta(@core, "to_a")

  def grad():
    result = @core.grad
    if null === result then null else Tensor(result)

  def grad!(gradient):
    handle = if gradient == null then null else gradient._core
    mutate!(@core, "set_grad", handle)
    self

  def copy(): Tensor(unary(@core, "copy", null))
  def snapshot(): Tensor(unary(@core, "snapshot", null))
  def detach(): Tensor(unary(@core, "detach", null))
  def to(device:, dtype: "same"): Tensor(transfer(@core, device_text(device), dtype.to_str))
  def to_str(): "Tensor(shape: #{shape}, dtype: #{dtype}, device: #{device})"

  def add(other): Tensor(binary(@core, _operand(other), "add"))
  def sub(other): Tensor(binary(@core, _operand(other), "sub"))
  def mul(other): Tensor(binary(@core, _operand(other), "mul"))
  def div(other): Tensor(binary(@core, _operand(other), "div"))
  def matmul(other): Tensor(binary(@core, other._core, "matmul"))
  def eq(other): Tensor(binary(@core, _operand(other), "eq"))

  def sum(dim: null, keepdim: false):
    Tensor(unary(@core, if dim == null then "sum" else "sum_dim", [dim, keepdim]))

  def mean(dim: null, keepdim: false):
    Tensor(unary(@core, if dim == null then "mean" else "mean_dim", [dim, keepdim]))

  def argmax(dim: null, keepdim: false): Tensor(unary(@core, "argmax", [dim, keepdim]))
  def index_select(dim, indices): Tensor(unary(@core, "index_select", [dim, indices._core]))

  def square(): Tensor(unary(@core, "square", null))
  def relu(): Tensor(unary(@core, "relu", null))
  def sigmoid(): Tensor(unary(@core, "sigmoid", null))
  def tanh(): Tensor(unary(@core, "tanh", null))

  def +(other): add(other)
  def -(other): sub(other)
  def *(other): mul(other)
  def /(other): div(other)
  def **(other): pow(other)

  def pow(other): Tensor(binary(@core, _operand(other), "pow"))
  def ne(other): Tensor(binary(@core, _operand(other), "ne"))
  def lt(other): Tensor(binary(@core, _operand(other), "lt"))
  def le(other): Tensor(binary(@core, _operand(other), "le"))
  def gt(other): Tensor(binary(@core, _operand(other), "gt"))
  def ge(other): Tensor(binary(@core, _operand(other), "ge"))

  def neg(): Tensor(unary(@core, "neg", null))
  def abs(): Tensor(unary(@core, "abs", null))
  def exp(): Tensor(unary(@core, "exp", null))
  def log(): Tensor(unary(@core, "log", null))
  def sqrt(): Tensor(unary(@core, "sqrt", null))
  def rsqrt(): Tensor(unary(@core, "rsqrt", null))
  def sin(): Tensor(unary(@core, "sin", null))
  def cos(): Tensor(unary(@core, "cos", null))

  def clamp(min:, max:): Tensor(unary(@core, "clamp", [min, max]))
  def softmax(dim: -1): Tensor(unary(@core, "softmax", dim))
  def log_softmax(dim: -1): Tensor(unary(@core, "log_softmax", dim))

  def flatten(start_dim: 0, end_dim: -1): Tensor(unary(@core, "flatten", [start_dim, end_dim]))
  def permute(dims): Tensor(unary(@core, "permute", dims))
  def unsqueeze(dim): Tensor(unary(@core, "unsqueeze", dim))
  def squeeze(dim: null): Tensor(unary(@core, "squeeze", dim))
  def contiguous(): Tensor(unary(@core, "contiguous", null))
  def [](index): select(0, index)
  def narrow(dim, start, length): Tensor(unary(@core, "narrow", [dim, start, length]))

  def split(size, dim: 0):
    raise ValueError("split size must be positive") if size <= 0
    length = shape[dim]
    return [narrow(dim, 0, 0)] if length == 0
    Array.of((length + size - 1) // size) |index|:
      start = index * size
      narrow(dim, start, Math.min(size, length - start))

  def chunk(chunks, dim: 0):
    raise ValueError("chunks must be positive") if chunks <= 0
    length = shape[dim]
    if length == 0:
      return Array.of(chunks): narrow(dim, 0, 0)
    split((length + chunks - 1) // chunks, dim:)

  def reshape(shape): Tensor(unary(@core, "reshape", shape))
  def transpose(dim0, dim1): Tensor(unary(@core, "transpose", [dim0, dim1]))
  def select(dim, index): Tensor(unary(@core, "select", [dim, index]))

  def add!(other):
    mutate!(@core, "add", _operand(other))
    self

  def sub!(other):
    mutate!(@core, "sub", _operand(other))
    self

  def mul!(other):
    mutate!(@core, "mul", _operand(other))
    self

  def fill!(value):
    mutate!(@core, "fill", value)
    self

  def zero!():
    mutate!(@core, "zero", null)
    self

  def copy_from!(other):
    mutate!(@core, "copy_from", other._core)
    self

  def requires_grad!(enabled):
    mutate!(@core, "requires_grad", enabled)
    self

  def zero_grad!():
    mutate!(@core, "zero_grad", true)
    self

  def zero_grad_buffers!():
    mutate!(@core, "zero_grad", false)
    self

  def backward!(gradient: null, retain_graph: false, create_graph: false):
    g = if null === gradient then null else gradient._core
    _backward!(@core, g, retain_graph, create_graph)
    self
