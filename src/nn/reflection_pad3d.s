package torch.nn.reflection_pad3d

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export ReflectionPad3d

class ReflectionPad3d < Module:
  def init(padding):
    _initialize_registry!
    @padding = if Int === padding then Array.filled(6, padding) else padding.array
    raise ValueError("expected 6 padding values") if @padding.count != 6

  def forward(x): kernel("pad", [x], [@padding, "reflect", 0.0])

