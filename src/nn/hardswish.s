package torch.nn.hardswish

from torch.nn.module import Module
from torch.nn.functional import hardswish

export Hardswish

class Hardswish < Module:
  def init(): _initialize_registry!
  def forward(x): hardswish(x)

