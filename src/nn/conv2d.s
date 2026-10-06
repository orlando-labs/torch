package sputniktorch.nn.conv2d

from sputniktorch.nn.convolution import Convolution

export Conv2d

class Conv2d < Convolution:
  def init(in_channels, out_channels, kernel_size, stride: 1, padding: 0,
      dilation: 1, groups: 1, bias: true, seed: 0, device: "cpu"):
    _initialize_convolution!(2, in_channels, out_channels, kernel_size,
      stride:, padding:, dilation:, groups:,
      bias:, seed:, device:)

