package torch.nn.lp_pool3d

from torch.nn.module import Module
from torch.nn.pooling_functional import lp_pool3d

export LPPool3d

class LPPool3d < Module:
  def init(@norm_type, @kernel_size, @stride: null, @ceil_mode: false): _initialize_registry!

  def forward(x): lp_pool3d(x, @norm_type, @kernel_size, stride: @stride, ceil_mode: @ceil_mode)
