package sputniktorch.nn.reflection_pad1d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export ReflectionPad1d

class ReflectionPad1d < Module:
  def init(padding):
    _initialize_registry!
    @padding = if Int === padding then Array.filled(2, padding) else padding.array
    raise ValueError("expected 2 padding values") if @padding.count != 2

  def forward(x): kernel("pad", [x], [@padding, "reflect", 0.0])

