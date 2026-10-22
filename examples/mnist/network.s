package torch.mnist.network

from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Sequential
from torch.nn import AdaptiveAvgPool2d, Flatten, Linear
from torch.mnist.residual_block import ResidualBlock

export SmallResNet

# Six convolutions, two residual blocks and global average pooling.
class SmallResNet < Module:
  def init(seed: 7, device: "cpu"):
    _initialize_registry!
    @features = Sequential([
      Conv2d(1, 16, 3, padding: 1, bias: false, seed:, device:),
      BatchNorm2d(16, device:),
      ReLU(),
      ResidualBlock(16, 16, seed: seed + 1, device:),
      ResidualBlock(16, 32, stride: 2, seed: seed + 4, device:),
      AdaptiveAvgPool2d(1),
      Flatten()
    ])
    @classifier = Linear(32, 10, seed: seed + 7, device:)

  def forward(x): @classifier(@features(x))
  def forward!(x): @classifier(@features.call!(x))
