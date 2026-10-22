package torch.nn.elu

from torch.nn.module import Module
from torch.nn.functional import elu

export ELU

class ELU < Module:
  def init(@alpha: 1.0): _initialize_registry!
  def forward(x): elu(x, alpha: @alpha)

