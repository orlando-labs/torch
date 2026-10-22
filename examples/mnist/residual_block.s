package torch.mnist.residual_block

from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Sequential, Identity

export ResidualBlock

class ResidualBlock < Module:
  def init(in_channels, out_channels, stride: 1, seed: 0, device: "cpu"):
    _initialize_registry!
    @body = Sequential([
      Conv2d(in_channels, out_channels, 3, stride:, padding: 1, bias: false, seed:, device:),
      BatchNorm2d(out_channels, device:),
      ReLU(),
      Conv2d(out_channels, out_channels, 3, padding: 1, bias: false, seed: seed + 1, device:),
      BatchNorm2d(out_channels, device:)
    ])
    @shortcut = if in_channels == out_channels and stride == 1:
      Identity()
    else:
      Sequential([
        Conv2d(in_channels, out_channels, 1, stride:, bias: false, seed: seed + 2, device:),
        BatchNorm2d(out_channels, device:)
      ])

  def forward(x): (@body(x) + @shortcut(x)).relu
  def forward!(x): (@body.call!(x) + @shortcut.call!(x)).relu
