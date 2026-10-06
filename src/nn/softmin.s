package sputniktorch.nn.softmin

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel, spatial

export Softmin

class Softmin < Module:
  def init(@dim: -1): _initialize_registry!
  def forward(x): x.neg.softmax(dim: @dim)

