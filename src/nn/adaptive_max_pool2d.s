package sputniktorch.nn.adaptive_max_pool2d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export AdaptiveMaxPool2d

class AdaptiveMaxPool2d < Module:
  def init(output_size):
    _initialize_registry!
    @options = [spatial(output_size, 2)]

  def forward(x): kernel("adaptive_max_pool2d", [x], @options)

