package torch.nn.softsign

from torch.nn.module import Module
from torch.nn.functional import softsign

export Softsign

class Softsign < Module:
  def init(): _initialize_registry!
  def forward(x): softsign(x)

