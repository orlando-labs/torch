package sputniktorch.nn.replication_pad3d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export ReplicationPad3d

class ReplicationPad3d < Module:
  def init(padding):
    _initialize_registry!
    @padding = if Int === padding then Array.filled(6, padding) else padding.array
    raise ValueError("expected 6 padding values") if @padding.count != 6

  def forward(x): kernel("pad", [x], [@padding, "replicate", 0.0])

