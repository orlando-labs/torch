package sputniktorch.nn.softsign

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import softsign

export Softsign

class Softsign < Module:
  def init(): _initialize_registry!
  def forward(x): softsign(x)

