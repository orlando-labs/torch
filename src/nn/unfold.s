package sputniktorch.nn.unfold

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export Unfold

class Unfold < Module:
  def init(kernel_size, dilation: 1, padding: 0, stride: 1):
    _initialize_registry!
    @options = [spatial(kernel_size, 2), spatial(dilation, 2),
      spatial(padding, 2), spatial(stride, 2)]

  def forward(x): kernel("unfold", [x], @options)
