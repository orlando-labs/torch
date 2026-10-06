package sputniktorch.onnx_tied_model

import sputniktorch
from sputniktorch.nn import Module

export TiedModel

class TiedModel < Module:
  attr weight

  def init(device):
    _initialize_registry!
    @weight = sputniktorch.ones([3], device:, requires_grad: true)
    register_parameter!("first", @weight)
    register_parameter!("second", @weight)

  def forward(x): x * @weight + @weight
