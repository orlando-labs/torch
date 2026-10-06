package sputniktorch.nn.pixel_unshuffle

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export PixelUnshuffle

class PixelUnshuffle < Module:
  def init(@factor): _initialize_registry!
  def forward(x): kernel("pixel_unshuffle", [x], [@factor])

