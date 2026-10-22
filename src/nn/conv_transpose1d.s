package torch.nn.conv_transpose1d

from torch.nn.convolution import Convolution

export ConvTranspose1d

class ConvTranspose1d < Convolution:
  def init(in_channels, out_channels, kernel_size, stride: 1, padding: 0,
      dilation: 1, groups: 1, bias: true, seed: 0, device: "cpu", output_padding: 0):
    _initialize_convolution!(1, in_channels, out_channels, kernel_size,
      stride:, padding:, dilation:, groups:,
      bias:, seed:, device:, transposed: true, output_padding:)

