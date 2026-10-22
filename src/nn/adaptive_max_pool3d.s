package torch.nn.adaptive_max_pool3d

from torch.nn.module import Module
from torch.nn.pooling_functional import adaptive_max_pool3d

export AdaptiveMaxPool3d

class AdaptiveMaxPool3d < Module:
  def init(@output_size, @return_indices: false):
    _initialize_registry!

  def forward(x): adaptive_max_pool3d(x, @output_size, return_indices: @return_indices)

