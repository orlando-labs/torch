package sputniktorch.nn.adaptive_max_pool3d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export AdaptiveMaxPool3d

class AdaptiveMaxPool3d < Module:
  def init(output_size):
    _initialize_registry!
    @options = [spatial(output_size, 3)]

  def forward(x): kernel("adaptive_max_pool3d", [x], @options)

