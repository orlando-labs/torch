package torch.nn.threshold

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export Threshold

class Threshold < Module:
  def init(@threshold, @value): _initialize_registry!
  def forward(x): kernel("threshold", [x], [@threshold, @value])

