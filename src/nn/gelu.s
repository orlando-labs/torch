package sputniktorch.nn.gelu

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import gelu

export GELU

class GELU < Module:
  def init(@approximate: :none): _initialize_registry!
  def forward(x): gelu(x, approximate: @approximate)

