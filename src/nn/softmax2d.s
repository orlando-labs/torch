package sputniktorch.nn.softmax2d

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export Softmax2d

class Softmax2d < Module:
  def init(): _initialize_registry!
  def forward(x): x.softmax(dim: -3)

