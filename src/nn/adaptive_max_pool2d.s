package torch.nn.adaptive_max_pool2d

from torch.nn.module import Module
from torch.nn.pooling_functional import adaptive_max_pool2d

export AdaptiveMaxPool2d

class AdaptiveMaxPool2d < Module:
  def init(@output_size, @return_indices: false):
    _initialize_registry!

  def forward(x): adaptive_max_pool2d(x, @output_size, return_indices: @return_indices)

