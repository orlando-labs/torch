package sputniktorch.nn.pixel_shuffle

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export PixelShuffle

class PixelShuffle < Module:
  def init(@factor): _initialize_registry!
  def forward(x): kernel("pixel_shuffle", [x], [@factor])

