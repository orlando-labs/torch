package sputniktorch.nn.replication_pad2d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export ReplicationPad2d

class ReplicationPad2d < Module:
  def init(padding):
    _initialize_registry!
    @padding = if Int === padding then Array.filled(4, padding) else padding.array
    raise ValueError("expected 4 padding values") if @padding.count != 4

  def forward(x): kernel("pad", [x], [@padding, "replicate", 0.0])

