package sputniktorch.nn.max_pool2d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export MaxPool2d

class MaxPool2d < Module:
  def init(kernel_size, stride: null, padding: 0, dilation: 1, ceil_mode: false):
    _initialize_registry!
    step = if stride == null then kernel_size else stride
    @options = [spatial(kernel_size, 2), spatial(step, 2),
      spatial(padding, 2), spatial(dilation, 2), ceil_mode]

  def forward(x): kernel("max_pool2d", [x], @options)

