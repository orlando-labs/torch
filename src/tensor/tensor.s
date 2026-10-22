package torch.tensor.value

from torch.tensor.core import meta, unary, binary, transfer, mutate!, backward! as _backward!
from torch.autograd.context import execution_mode
from torch.tensor.core import make as _make
from torch.tensor.indexing_core import indexing as _indexing
from torch.tensor.slice import Slice
from torch.device import Device, device_text
from torch.dtype import DType
from torch.errors import DTypeError

export Tensor

def _operand(value):
  if Tensor === value:
    value._core
  else:
    value

def _index_operand(value):
  if Tensor === value:
    value._core
  elif Slice === value:
    value._arguments
  elif Array === value:
    dtype = if value.count == 0 then "int64" else "auto"
    _make(value, dtype, "cpu", false)
  elif Symbol === value:
    return [null, null, 1] if value == :all
    return "..." if value == :ellipsis
    raise TypeError("unknown tensor index Symbol")
  else:
    value

def _indices(values): values.array.map: _index_operand($it)

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
  def is_sparse(): meta(@core, "sparse")
  def sparse?(): is_sparse
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
  def to_dense(): Tensor(unary(@core, "to_dense", null))
  def densified(): to_dense
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
  def relu(): Tensor(@core.relu(execution_mode()))
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
  def real(): Tensor(unary(@core, "real", null))
  def imag(): Tensor(unary(@core, "imag", null))
  def conj(): Tensor(unary(@core, "conj", null))
  def resolve_conj(): Tensor(unary(@core, "resolve_conj", null))
  def angle(): Tensor(unary(@core, "angle", null))
  def view_as_real(): Tensor(unary(@core, "view_as_real", null))
  def view_as_complex(): Tensor(unary(@core, "view_as_complex", null))

  def clamp(min:, max:): Tensor(unary(@core, "clamp", [min, max]))
  def softmax(dim: -1): Tensor(unary(@core, "softmax", dim))
  def log_softmax(dim: -1): Tensor(unary(@core, "log_softmax", dim))

  def flatten(start_dim: 0, end_dim: -1): Tensor(unary(@core, "flatten", [start_dim, end_dim]))
  def permute(dims): Tensor(unary(@core, "permute", dims))
  def unsqueeze(dim): Tensor(unary(@core, "unsqueeze", dim))
  def squeeze(dim: null): Tensor(unary(@core, "squeeze", dim))
  def contiguous(): Tensor(unary(@core, "contiguous", null))

  def [](key):
    return select(0, key) if Int === key
    index(if Tuple === key then key.array else [key])

  def []=(key, value):
    index_put!(if Tuple === key then key.array else [key], value)

  def index(indices): Tensor(_indexing(@core, "index", [], [_indices(indices)]))

  def index_put(indices, values, accumulate: false):
    Tensor(_indexing(@core, "index_put", [_operand(values)], [_indices(indices), accumulate]))

  def index_put!(indices, values, accumulate: false):
    _indexing(@core, "index_put", [_operand(values)], [_indices(indices), accumulate], mutating: true)
    self

  def gather(dim, index, sparse_grad: false):
    Tensor(_indexing(@core, "gather", [index._core], [dim, sparse_grad]))

  def take(index):
    raise DTypeError("take requires int64 indices") unless index.dtype.to_str == "int64"
    if device.type == "mps":
      result = reshape([-1]).index([index.reshape([-1])])
      return result.reshape(index.shape)
    Tensor(_indexing(@core, "take", [index._core], []))

  def take_along_dim(indices, dim: null):
    Tensor(_indexing(@core, "take_along_dim", [indices._core], [dim]))

  def put(index, source, accumulate: false):
    return _put_mps(index, source, accumulate, false) if device.type == "mps"
    Tensor(_indexing(@core, "put", [index._core, source._core], [accumulate]))

  def put!(index, source, accumulate: false):
    return copy_from!(_put_mps(index, source, accumulate, true)) if device.type == "mps"
    _indexing(@core, "put", [index._core, source._core], [accumulate], mutating: true)
    self

  def _put_mps(index, source, accumulate, mutating):
    _indexing(@core, "put_prepare", [index._core, source._core], [], mutating:)
    result = reshape([-1]).index_put([index.reshape([-1])], source.reshape([-1]), accumulate:)
    result.reshape(shape)

  def scatter(dim, index, source, reduce: null):
    reduction = if reduce == null then null else reduce.str
    Tensor(_indexing(@core, "scatter", [index._core, _operand(source)], [dim, reduction]))

  def scatter!(dim, index, source, reduce: null):
    reduction = if reduce == null then null else reduce.str
    _indexing(@core, "scatter", [index._core, _operand(source)], [dim, reduction], mutating: true)
    self

  def scatter_add(dim, index, source):
    Tensor(_indexing(@core, "scatter_add", [index._core, source._core], [dim]))

  def scatter_add!(dim, index, source):
    _indexing(@core, "scatter_add", [index._core, source._core], [dim], mutating: true)
    self

  def scatter_reduce(dim, index, source, reduce, include_self: true):
    Tensor(_indexing(@core, "scatter_reduce", [index._core, source._core], [dim, reduce.str, include_self]))

  def scatter_reduce!(dim, index, source, reduce, include_self: true):
    _indexing(@core, "scatter_reduce", [index._core, source._core], [dim, reduce.str, include_self], mutating: true)
    self

  def index_add(dim, index, source, alpha: 1):
    Tensor(_indexing(@core, "index_add", [index._core, source._core], [dim, alpha]))

  def index_add!(dim, index, source, alpha: 1):
    _indexing(@core, "index_add", [index._core, source._core], [dim, alpha], mutating: true)
    self

  def index_copy(dim, index, source):
    Tensor(_indexing(@core, "index_copy", [index._core, source._core], [dim]))

  def index_copy!(dim, index, source):
    _indexing(@core, "index_copy", [index._core, source._core], [dim], mutating: true)
    self

  def index_fill(dim, index, value):
    Tensor(_indexing(@core, "index_fill", [index._core, _operand(value)], [dim]))

  def index_fill!(dim, index, value):
    _indexing(@core, "index_fill", [index._core, _operand(value)], [dim], mutating: true)
    self

  def index_reduce(dim, index, source, reduce, include_self: true):
    Tensor(_indexing(@core, "index_reduce", [index._core, source._core], [dim, reduce.str, include_self]))

  def index_reduce!(dim, index, source, reduce, include_self: true):
    _indexing(@core, "index_reduce", [index._core, source._core], [dim, reduce.str, include_self], mutating: true)
    self

  def masked_select(mask): Tensor(_indexing(@core, "masked_select", [mask._core], []))

  def masked_fill(mask, value):
    Tensor(_indexing(@core, "masked_fill", [mask._core, _operand(value)], []))

  def masked_fill!(mask, value):
    _indexing(@core, "masked_fill", [mask._core, _operand(value)], [], mutating: true)
    self

  def masked_scatter(mask, source):
    Tensor(_indexing(@core, "masked_scatter", [mask._core, source._core], []))

  def masked_scatter!(mask, source):
    _indexing(@core, "masked_scatter", [mask._core, source._core], [], mutating: true)
    self

  def nonzero(as_tuple: false):
    result = _indexing(@core, "nonzero", [], [as_tuple])
    if as_tuple then (result.map: Tensor($it)).tuple else Tensor(result)

  def select_scatter(source, dim, index):
    Tensor(_indexing(@core, "select_scatter", [source._core], [dim, index]))

  def slice_scatter(source, dim: 0, start: null, end: null, step: 1):
    Tensor(_indexing(@core, "slice_scatter", [source._core], [dim, start, end, step]))

  def diagonal_scatter(source, offset: 0, dim1: 0, dim2: 1):
    Tensor(_indexing(@core, "diagonal_scatter", [source._core], [offset, dim1, dim2]))

  def indexed(indices): index(indices)
  def gathered(dim, index, sparse_grad: false): gather(dim, index, sparse_grad:)
  def scattered(dim, index, source, reduce: null): scatter(dim, index, source, reduce:)
  def scatter_added(dim, index, source): scatter_add(dim, index, source)

  def scatter_reduced(dim, index, source, reduce, include_self: true):
    scatter_reduce(dim, index, source, reduce, include_self:)

  def index_added(dim, index, source, alpha: 1): index_add(dim, index, source, alpha:)
  def index_copied(dim, index, source): index_copy(dim, index, source)
  def index_filled(dim, index, value): index_fill(dim, index, value)

  def index_reduced(dim, index, source, reduce, include_self: true):
    index_reduce(dim, index, source, reduce, include_self:)

  def masked_selected(mask): masked_select(mask)
  def masked_filled(mask, value): masked_fill(mask, value)
  def masked_scattered(mask, source): masked_scatter(mask, source)
  def index_selected(dim, indices): index_select(dim, indices)
  def selected(dim, index): select(dim, index)

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
