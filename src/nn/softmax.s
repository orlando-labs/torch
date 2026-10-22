package torch.nn.softmax

from torch.nn.module import Module
from torch.nn.functional import softmax

export Softmax

class Softmax < Module:
  def init(@dim: -1): _initialize_registry!
  def forward(x): softmax(x, dim: @dim)

