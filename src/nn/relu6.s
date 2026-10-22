package torch.nn.relu6

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export ReLU6

class ReLU6 < Module:
  def init(): _initialize_registry!
  def forward(x): kernel("hardtanh", [x], [0.0, 6.0])

