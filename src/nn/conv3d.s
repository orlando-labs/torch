package torch.nn.conv3d

from torch.nn.convolution import Convolution

export Conv3d

class Conv3d < Convolution:
  def init(in_channels, out_channels, kernel_size, stride: 1, padding: 0,
      dilation: 1, groups: 1, bias: true, seed: 0, device: "cpu"):
    _initialize_convolution!(3, in_channels, out_channels, kernel_size,
      stride:, padding:, dilation:, groups:,
      bias:, seed:, device:)

