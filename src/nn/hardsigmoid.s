package torch.nn.hardsigmoid

from torch.nn.module import Module
from torch.nn.functional import hardsigmoid

export Hardsigmoid

class Hardsigmoid < Module:
  def init(): _initialize_registry!
  def forward(x): hardsigmoid(x)

