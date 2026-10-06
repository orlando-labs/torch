package sputniktorch.nn.sigmoid

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import sigmoid

export Sigmoid

class Sigmoid < Module:
  def init(): _initialize_registry!
  def forward(x): sigmoid(x)

