package torch.nn.replication_pad1d

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export ReplicationPad1d

class ReplicationPad1d < Module:
  def init(padding):
    _initialize_registry!
    @padding = if Int === padding then Array.filled(2, padding) else padding.array
    raise ValueError("expected 2 padding values") if @padding.count != 2

  def forward(x): kernel("pad", [x], [@padding, "replicate", 0.0])

