package torch.nn.silu

from torch.nn.module import Module
from torch.nn.functional import silu

export SiLU

class SiLU < Module:
  def init(): _initialize_registry!
  def forward(x): silu(x)

