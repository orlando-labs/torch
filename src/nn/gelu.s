package torch.nn.gelu

from torch.nn.module import Module
from torch.nn.functional import gelu

export GELU

class GELU < Module:
  def init(@approximate: :none): _initialize_registry!
  def forward(x): gelu(x, approximate: @approximate)

