package sputniktorch.nn.max_pool3d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export MaxPool3d

class MaxPool3d < Module:
  def init(kernel_size, stride: null, padding: 0, dilation: 1, ceil_mode: false):
    _initialize_registry!
    step = if stride == null then kernel_size else stride
    @options = [spatial(kernel_size, 3), spatial(step, 3),
      spatial(padding, 3), spatial(dilation, 3), ceil_mode]

  def forward(x): kernel("max_pool3d", [x], @options)

