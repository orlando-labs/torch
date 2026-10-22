package torch.nn.pixel_unshuffle

from torch.nn.module import Module
from torch.nn.functional import kernel, spatial

export PixelUnshuffle

class PixelUnshuffle < Module:
  def init(@factor): _initialize_registry!
  def forward(x): kernel("pixel_unshuffle", [x], [@factor])

