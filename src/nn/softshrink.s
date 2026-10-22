package torch.nn.softshrink

from torch.nn.module import Module
from torch.nn.functional import softshrink

export Softshrink

class Softshrink < Module:
  def init(@threshold: 0.5): _initialize_registry!
  def forward(x): softshrink(x, threshold: @threshold)

