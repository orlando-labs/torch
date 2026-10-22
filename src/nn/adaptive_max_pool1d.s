package torch.nn.adaptive_max_pool1d

from torch.nn.module import Module
from torch.nn.pooling_functional import adaptive_max_pool1d

export AdaptiveMaxPool1d

class AdaptiveMaxPool1d < Module:
  def init(@output_size, @return_indices: false):
    _initialize_registry!

  def forward(x): adaptive_max_pool1d(x, @output_size, return_indices: @return_indices)

