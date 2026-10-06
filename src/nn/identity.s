package sputniktorch.nn.identity

from sputniktorch.nn.module import Module

export Identity

class Identity < Module:
  def init(): _initialize_registry!
  def forward(x): x
