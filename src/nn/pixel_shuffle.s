package torch.nn.pixel_shuffle

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export PixelShuffle

class PixelShuffle < Module:
  def init(@factor): _initialize_registry!
  def forward(x): kernel("pixel_shuffle", [x], [@factor])

