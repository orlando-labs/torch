package sputniktorch.nn.prelu

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel
from sputniktorch.tensor.factories import ones

export PReLU

class PReLU < Module:
  attr weight

  def init(num_parameters: 1, initial: 0.25, device: "cpu"):
    _initialize_registry!
    @weight = (ones([num_parameters], device:) * initial).requires_grad!(true)
    register_parameter!("weight", @weight)

  def forward(x): kernel("prelu", [x, @weight])
