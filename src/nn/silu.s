package sputniktorch.nn.silu

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import silu

export SiLU

class SiLU < Module:
  def init(): _initialize_registry!
  def forward(x): silu(x)

