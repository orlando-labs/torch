package torch.nn.conv_transpose2d

from torch.nn.convolution import Convolution

export ConvTranspose2d

class ConvTranspose2d < Convolution:
  def init(in_channels, out_channels, kernel_size, stride: 1, padding: 0,
      dilation: 1, groups: 1, bias: true, seed: 0, device: "cpu", output_padding: 0):
    _initialize_convolution!(2, in_channels, out_channels, kernel_size,
      stride:, padding:, dilation:, groups:,
      bias:, seed:, device:, transposed: true, output_padding:)

