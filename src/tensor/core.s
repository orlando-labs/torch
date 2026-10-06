package sputniktorch.tensor.core

from sputniktorch.errors import Error
from sputniktorch.autograd.context import execution_mode

export TensorCore, make, factory, meta, binary, unary, transfer, mutate!, backward!

native class TensorCore from "sputniktorch.TensorCore" collected:
  def destroy!() from "amt_tensor_free"
  def shape() from "amt_tensor_shape"
  def strides() from "amt_tensor_strides"
  def rank() from "amt_tensor_rank"
  def numel() from "amt_tensor_numel"
  def dtype() from "amt_tensor_dtype"
  def device() from "amt_tensor_device"
  def requires_grad?() from "amt_tensor_requires_grad"
  def leaf?() from "amt_tensor_leaf"
  def readonly?() from "amt_tensor_readonly"
  def contiguous?() from "amt_tensor_contiguous"
  def grad() from "amt_tensor_grad"
  def identity() from "amt_tensor_identity"

native def _make_tensor(data, dtype as Str, device as Str, requires_grad as Bool, mode as Int) from "amt_tensor"

native def _factory_tensor(op as Str, shape, dtype as Str, device as Str,
  requires_grad as Bool, options, mode as Int) from "amt_factory"

native def _meta(core, key as Str, mode as Int) from "amt_meta"
native def _materialize(core, key as Str, mode as Int) from "amt_materialize"
native def _binary(core, other, op as Str, mode as Int) from "amt_binary"
native def _unary(core, op as Str, args, mode as Int) from "amt_unary"
native def _transfer(core, device as Str, dtype as Str, mode as Int) from "amt_to"
native def _mutate!(core, op as Str, value, mode as Int) from "amt_mutate"
native def _backward!(core, gradient, retain_graph as Bool, create_graph as Bool, mode as Int) from "amt_backward"

def binary(core, other, op): _binary(core, other, op, execution_mode())
def meta(core, key):
  mode = execution_mode()
  if key == "item" or key == "to_a":
    _materialize(core, key, mode)
  else:
    _meta(core, key, mode)

def unary(core, op, args): _unary(core, op, args, execution_mode())
def transfer(core, device, dtype): _transfer(core, device, dtype, execution_mode())
def mutate!(core, op, value): _mutate!(core, op, value, execution_mode())

def backward!(core, gradient, retain_graph, create_graph):
  _backward!(core, gradient, retain_graph, create_graph, execution_mode())

def make(data, dtype, device, requires_grad): _make_tensor(data, dtype, device, requires_grad, execution_mode())

def factory(op, shape, dtype, device, requires_grad, options):
  _factory_tensor(op, shape, dtype, device, requires_grad, options, execution_mode())
