package torch.nn.max_unpool2d

from torch.nn.module import Module
from torch.nn.pooling_functional import max_unpool2d

export MaxUnpool2d

class MaxUnpool2d < Module:
  def init(@kernel_size, @stride: null, @padding: 0): _initialize_registry!

  def forward(x, indices, output_size: null):
    max_unpool2d(x, indices, @kernel_size, stride: @stride, padding: @padding, output_size:)
