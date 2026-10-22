package torch.nn.softmax2d

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export Softmax2d

class Softmax2d < Module:
  def init(): _initialize_registry!
  def forward(x): x.softmax(dim: -3)

