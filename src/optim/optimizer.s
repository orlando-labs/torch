package torch.optim.optimizer

from torch.errors import Error
from torch.tensor.value import Tensor
from torch.autograd import no_grad
from torch.backends import execution_batch
from torch.tensor.foreach import zero_grad! as _zero_grad!

export Optimizer, validate_coefficients

native def _validate(parameters, lr) from "amt_optimizer_validate"
native def validate_coefficients(values) from "amt_optimizer_coefficients"

class Optimizer:
  attr lr

  def _initialize_optimizer!(parameters, lr):
    @parameters = parameters.uniq: $it._identity
    @cores = @parameters.map: $it._core
    @lr = lr
    _validate_step
    self

  def _validate_step():
    _validate(@cores, @lr)

  def _run_step!(&block):
    operation = ||:
      _validate_step
      no_grad(&block)
    execution_batch(&operation)

  def parameters(): @parameters.copy

  def _validate_slots(slots):
    raise ValueError("optimizer slot count mismatch") if slots.count != @parameters.count
    slots.each_with_index |slot, index|:
      if slot == null:
        next null
      raise TypeError("optimizer slots must be Tensors") unless (Tensor === slot)
      parameter = @parameters[index]
      if slot.shape != parameter.shape or slot.dtype != parameter.dtype:
        raise ValueError("optimizer slot shape or dtype mismatch")
    self

  def lr!(value):
    _validate(@cores, value)
    @lr = value
    self

  def zero_grad!():
    _zero_grad!(@cores, true)
    self

  def zero_grad_buffers!():
    _zero_grad!(@cores, false)
    self
