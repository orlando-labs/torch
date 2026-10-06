package sputniktorch.nn.mish

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import mish

export Mish

class Mish < Module:
  def init(): _initialize_registry!
  def forward(x): mish(x)

