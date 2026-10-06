package sputniktorch.nn.hardsigmoid

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import hardsigmoid

export Hardsigmoid

class Hardsigmoid < Module:
  def init(): _initialize_registry!
  def forward(x): hardsigmoid(x)

