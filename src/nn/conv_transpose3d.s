package torch.nn.conv_transpose3d

from torch.nn.convolution import Convolution

export ConvTranspose3d

class ConvTranspose3d < Convolution:
  def init(in_channels, out_channels, kernel_size, stride: 1, padding: 0,
      dilation: 1, groups: 1, bias: true, seed: 0, device: "cpu", output_padding: 0):
    _initialize_convolution!(3, in_channels, out_channels, kernel_size,
      stride:, padding:, dilation:, groups:,
      bias:, seed:, device:, transposed: true, output_padding:)

