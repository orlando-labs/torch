package torch.nn.module

from torch.errors import ShapeError
from torch.autograd.context import execution_mode
from torch.tensor.value import Tensor

export Module

class Module:
  def init(): _initialize_registry!

  def _initialize_registry!():
    @parameters = {}
    @modules = {}
    @buffers = {}
    @training = true

  def after_init!():
    _initialize_registry! if @modules == null
    # Explicit registrations take precedence, including aliases whose names
    # differ from the field name. Snapshot before discovering new children so
    # multiple automatic fields can still refer to the same child.
    registered = @modules.values
    instance_fields.each |name, child|:
      register_module!(name, (&child)) if (Module === &child) and not registered.include?((&child))
    self

  attr training? from @training

  def train!(enabled: true):
    @training = enabled
    @modules.each |name, child|: child.train!(enabled:)
    self

  def eval!(): train!(enabled: false)
  def stateful?():
    if @modules.empty?:
      false
    else:
      @modules.values.any?: $it.stateful?

  def _validate_member(name):
    if not (Str === name) or name == "" or name.split(".").count != 1:
      raise ValueError("module member names must be nonempty strings without dots")
    if @parameters.has_key?(name) or @modules.has_key?(name) or @buffers.has_key?(name):
      raise ValueError("duplicate module member: #{name}")

  def _contains_module(mod): self == mod or @modules.values.any?: $it._contains_module(mod)

  def register_parameter!(name, parameter):
    _validate_member(name)
    raise TypeError("parameter must be a Tensor") unless (Tensor === parameter)
    raise ValueError("parameter must be a leaf") unless parameter.leaf?
    @parameters[name] = parameter
    self

  def register_module!(name, mod):
    _validate_member(name)
    raise TypeError("child must be a Module") unless (Module === mod)
    raise ValueError("module registration would create a cycle") if mod._contains_module(self)
    mod.train!(enabled: @training)
    @modules[name] = mod
    self

  def register_buffer!(name, value):
    _validate_member(name)
    raise TypeError("buffer must be a Tensor") unless (Tensor === value)
    @buffers[name] = value
    self

  def named_parameters():
    result = @parameters.copy
    @modules.each |name, mod|:
      mod.named_parameters.each |key, value|:
        result[name + "." + key] = value
    result

  def parameters(): named_parameters.values

  def named_buffers():
    result = @buffers.copy
    @modules.each |name, child|:
      child.named_buffers.each |key, value|:
        result["#{name}.#{key}"] = value
    result

  def state_dict(): named_parameters.merge(named_buffers).transformed_values: $it.snapshot

  def load_state_dict!(state):
    named = named_parameters.merge(named_buffers)
    if state.count != named.count:
      raise ShapeError("state dict key count mismatch")
    named.each |key, p|:
      raise ValueError("state target is read-only: #{key}") if p.readonly?
      if not state.has_key?(key):
        raise ShapeError("missing state dict key")
      if p.shape != state[key].shape or p.dtype != state[key].dtype:
        raise ShapeError("state dict shape or dtype mismatch")
    # Native parameter loading uses a dedicated no-grad copy operation.
    mode = execution_mode()
    named.each |key, p|: _load_parameter!(p._core, state[key]._core, mode)
    self

  def call(*inputs, **options):
    if stateful?:
      raise ValueError("training forward changes state; use call! or eval! first")
    forward(*inputs, **options)

  def call!(*inputs, **options): forward!(*inputs, **options)
  def forward!(*inputs, **options): forward(*inputs, **options)

native def _load_parameter!(target, source, mode as Int) from "amt_load_parameter" blocking
