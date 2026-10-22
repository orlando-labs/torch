package torch.nn.reflection_pad2d

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export ReflectionPad2d

class ReflectionPad2d < Module:
  def init(padding):
    _initialize_registry!
    @padding = if Int === padding then Array.filled(4, padding) else padding.array
    raise ValueError("expected 4 padding values") if @padding.count != 4

  def forward(x): kernel("pad", [x], [@padding, "reflect", 0.0])

