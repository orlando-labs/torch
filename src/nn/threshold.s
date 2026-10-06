package sputniktorch.nn.threshold

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export Threshold

class Threshold < Module:
  def init(@threshold, @value): _initialize_registry!
  def forward(x): kernel("threshold", [x], [@threshold, @value])

