package torch.nn.constant_pad1d

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export ConstantPad1d

class ConstantPad1d < Module:
  def init(padding, @value = 0.0):
    _initialize_registry!
    @padding = if Int === padding then Array.filled(2, padding) else padding.array
    raise ValueError("expected 2 padding values") if @padding.count != 2

  def forward(x): kernel("pad", [x], [@padding, "constant", @value])

