package sputniktorch.nn.glu

from sputniktorch.nn.module import Module

export GLU

class GLU < Module:
  def init(@dim: -1): _initialize_registry!

  def forward(x):
    raise ValueError("GLU axis must have an even size") if x.shape[@dim] % 2 != 0
    [values, gates] = x.chunk(2, dim: @dim)
    values * gates.sigmoid
