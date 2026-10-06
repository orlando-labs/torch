package sputniktorch.nn.bilinear

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel
from sputniktorch.tensor.factories import randn, zeros

export Bilinear

class Bilinear < Module:
  attr weight
  attr bias

  def init(in1_features, in2_features, out_features, bias: true, seed: 0, device: "cpu"):
    _initialize_registry!
    if in1_features <= 0 or in2_features <= 0 or out_features <= 0:
      raise ValueError("feature counts must be positive")
    scale = (in1_features * in2_features).float ** 0.5
    shape = [out_features, in1_features, in2_features]
    @weight = (randn(shape, seed:, device:) / scale).requires_grad!(true)
    @bias = if bias then zeros([out_features], device:, requires_grad: true) else null
    register_parameter!("weight", @weight)
    register_parameter!("bias", @bias) if bias

  def forward(first, second): kernel("bilinear", [first, second, @weight, @bias])
