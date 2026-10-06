package sputniktorch.nn.selu

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import selu

export SELU

class SELU < Module:
  def init(): _initialize_registry!
  def forward(x): selu(x)

