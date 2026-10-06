package sputniktorch.nn.softshrink

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import softshrink

export Softshrink

class Softshrink < Module:
  def init(@threshold: 0.5): _initialize_registry!
  def forward(x): softshrink(x, threshold: @threshold)

