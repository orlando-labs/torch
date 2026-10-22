package torch.nn.tanhshrink

from torch.nn.module import Module
from torch.nn.functional import tanhshrink

export Tanhshrink

class Tanhshrink < Module:
  def init(): _initialize_registry!
  def forward(x): tanhshrink(x)

