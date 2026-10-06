package sputniktorch.nn.tanh

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import tanh

export Tanh

class Tanh < Module:
  def init(): _initialize_registry!
  def forward(x): tanh(x)

