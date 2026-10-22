package torch.nn.softmin

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export Softmin

class Softmin < Module:
  def init(@dim: -1): _initialize_registry!
  def forward(x): x.neg.softmax(dim: @dim)

