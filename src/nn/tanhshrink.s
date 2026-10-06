package sputniktorch.nn.tanhshrink

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import tanhshrink

export Tanhshrink

class Tanhshrink < Module:
  def init(): _initialize_registry!
  def forward(x): tanhshrink(x)

