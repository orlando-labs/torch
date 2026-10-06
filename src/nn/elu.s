package sputniktorch.nn.elu

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import elu

export ELU

class ELU < Module:
  def init(@alpha: 1.0): _initialize_registry!
  def forward(x): elu(x, alpha: @alpha)

