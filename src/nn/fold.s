package torch.nn.fold

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export Fold

class Fold < Module:
  def init(output_size, kernel_size, dilation: 1, padding: 0, stride: 1):
    _initialize_registry!
    @options = [spatial(output_size, 2), spatial(kernel_size, 2),
      spatial(dilation, 2), spatial(padding, 2), spatial(stride, 2)]

  def forward(x): kernel("fold", [x], @options)
