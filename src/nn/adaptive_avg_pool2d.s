package torch.nn.adaptive_avg_pool2d

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export AdaptiveAvgPool2d

class AdaptiveAvgPool2d < Module:
  def init(output_size):
    _initialize_registry!
    @options = [spatial(output_size, 2)]

  def forward(x): kernel("adaptive_avg_pool2d", [x], @options)

