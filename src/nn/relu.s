package sputniktorch.nn.relu

from sputniktorch.nn.module import Module

export ReLU

class ReLU < Module:
  def init(): _initialize_registry!
  def forward(x): x.relu
