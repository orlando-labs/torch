package sputniktorch.nn.avg_pool3d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export AvgPool3d

class AvgPool3d < Module:
  def init(kernel_size, stride: null, padding: 0, ceil_mode: false, count_include_pad: true):
    _initialize_registry!
    step = if stride == null then kernel_size else stride
    @options = [spatial(kernel_size, 3), spatial(step, 3),
      spatial(padding, 3), ceil_mode, count_include_pad]

  def forward(x): kernel("avg_pool3d", [x], @options)

