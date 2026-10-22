package torch.nn.tanh

from torch.nn.module import Module
from torch.nn.functional import tanh

export Tanh

class Tanh < Module:
  def init(): _initialize_registry!
  def forward(x): tanh(x)

