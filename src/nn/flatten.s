package sputniktorch.nn.flatten

from sputniktorch.nn.module import Module

export Flatten

class Flatten < Module:
  def init(@start_dim: 1, @end_dim: -1): _initialize_registry!
  def forward(x): x.flatten(start_dim: @start_dim, end_dim: @end_dim)
