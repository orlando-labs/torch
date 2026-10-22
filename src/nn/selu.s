package torch.nn.selu

from torch.nn.module import Module
from torch.nn.functional import selu

export SELU

class SELU < Module:
  def init(): _initialize_registry!
  def forward(x): selu(x)

