package torch.nn.hardtanh

from torch.nn.module import Module
from torch.nn.functional import hardtanh

export Hardtanh

class Hardtanh < Module:
  def init(@min_value: -1.0, @max_value: 1.0): _initialize_registry!
  def forward(x): hardtanh(x, min_value: @min_value, max_value: @max_value)

