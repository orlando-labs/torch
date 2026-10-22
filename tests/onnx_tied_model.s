package torch.onnx_tied_model

import torch
from torch.nn import Module

export TiedModel

class TiedModel < Module:
  attr weight

  def init(device):
    _initialize_registry!
    @weight = torch.ones([3], device:, requires_grad: true)
    register_parameter!("first", @weight)
    register_parameter!("second", @weight)

  def forward(x): x * @weight + @weight
