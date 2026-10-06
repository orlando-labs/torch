package sputniktorch.nn.avg_pool1d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export AvgPool1d

class AvgPool1d < Module:
  def init(kernel_size, stride: null, padding: 0, ceil_mode: false, count_include_pad: true):
    _initialize_registry!
    step = if stride == null then kernel_size else stride
    @options = [spatial(kernel_size, 1), spatial(step, 1),
      spatial(padding, 1), ceil_mode, count_include_pad]

  def forward(x): kernel("avg_pool1d", [x], @options)

