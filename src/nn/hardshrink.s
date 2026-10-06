package sputniktorch.nn.hardshrink

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import hardshrink

export Hardshrink

class Hardshrink < Module:
  def init(@threshold: 0.5): _initialize_registry!
  def forward(x): hardshrink(x, threshold: @threshold)

