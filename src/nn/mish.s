package torch.nn.mish

from torch.nn.module import Module
from torch.nn.functional import mish

export Mish

class Mish < Module:
  def init(): _initialize_registry!
  def forward(x): mish(x)

