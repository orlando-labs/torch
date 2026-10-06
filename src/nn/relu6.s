package sputniktorch.nn.relu6

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export ReLU6

class ReLU6 < Module:
  def init(): _initialize_registry!
  def forward(x): kernel("hardtanh", [x], [0.0, 6.0])

