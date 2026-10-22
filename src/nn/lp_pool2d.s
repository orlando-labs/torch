package torch.nn.lp_pool2d

from torch.nn.module import Module
from torch.nn.pooling_functional import lp_pool2d

export LPPool2d

class LPPool2d < Module:
  def init(@norm_type, @kernel_size, @stride: null, @ceil_mode: false): _initialize_registry!

  def forward(x): lp_pool2d(x, @norm_type, @kernel_size, stride: @stride, ceil_mode: @ceil_mode)
