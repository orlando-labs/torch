package sputniktorch.nn.softplus

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import softplus

export Softplus

class Softplus < Module:
  def init(@beta: 1.0, @threshold: 20.0): _initialize_registry!
  def forward(x): softplus(x, beta: @beta, threshold: @threshold)

