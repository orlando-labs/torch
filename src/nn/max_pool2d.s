package torch.nn.max_pool2d

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial
from torch.nn.spatial_core import spatial_kernel

export MaxPool2d

class MaxPool2d < Module:
  def init(kernel_size, stride: null, padding: 0, dilation: 1, ceil_mode: false, @return_indices: false):
    _initialize_registry!
    step = if stride == null then kernel_size else stride
    @options = [spatial(kernel_size, 2), spatial(step, 2),
      spatial(padding, 2), spatial(dilation, 2), ceil_mode]

  def forward(x):
    if @return_indices:
      spatial_kernel("max_pool2d", [x], [*@options, true], pair: true)
    else:
      kernel("max_pool2d", [x], @options)

