package torch.nn.adaptive_avg_pool1d

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export AdaptiveAvgPool1d

class AdaptiveAvgPool1d < Module:
  def init(output_size):
    _initialize_registry!
    @options = [spatial(output_size, 1)]

  def forward(x): kernel("adaptive_avg_pool1d", [x], @options)

