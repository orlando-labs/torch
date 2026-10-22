package torch.nn.conv1d

from torch.nn.convolution import Convolution

export Conv1d

class Conv1d < Convolution:
  def init(in_channels, out_channels, kernel_size, stride: 1, padding: 0,
      dilation: 1, groups: 1, bias: true, seed: 0, device: "cpu"):
    _initialize_convolution!(1, in_channels, out_channels, kernel_size,
      stride:, padding:, dilation:, groups:,
      bias:, seed:, device:)

