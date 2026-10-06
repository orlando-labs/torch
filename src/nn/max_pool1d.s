package sputniktorch.nn.max_pool1d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export MaxPool1d

class MaxPool1d < Module:
  def init(kernel_size, stride: null, padding: 0, dilation: 1, ceil_mode: false):
    _initialize_registry!
    step = if stride == null then kernel_size else stride
    @options = [spatial(kernel_size, 1), spatial(step, 1),
      spatial(padding, 1), spatial(dilation, 1), ceil_mode]

  def forward(x): kernel("max_pool1d", [x], @options)

