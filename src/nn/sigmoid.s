package torch.nn.sigmoid

from torch.nn.module import Module
from torch.nn.functional import sigmoid

export Sigmoid

class Sigmoid < Module:
  def init(): _initialize_registry!
  def forward(x): sigmoid(x)

