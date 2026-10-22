package torch.nn.avg_pool2d

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export AvgPool2d

class AvgPool2d < Module:
  def init(kernel_size, stride: null, padding: 0, ceil_mode: false, count_include_pad: true):
    _initialize_registry!
    step = if stride == null then kernel_size else stride
    @options = [spatial(kernel_size, 2), spatial(step, 2),
      spatial(padding, 2), ceil_mode, count_include_pad]

  def forward(x): kernel("avg_pool2d", [x], @options)

