package sputniktorch.nn.adaptive_max_pool1d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export AdaptiveMaxPool1d

class AdaptiveMaxPool1d < Module:
  def init(output_size):
    _initialize_registry!
    @options = [spatial(output_size, 1)]

  def forward(x): kernel("adaptive_max_pool1d", [x], @options)

