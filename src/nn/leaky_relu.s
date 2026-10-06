package sputniktorch.nn.leaky_relu

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import leaky_relu

export LeakyReLU

class LeakyReLU < Module:
  def init(@negative_slope: 0.01): _initialize_registry!
  def forward(x): leaky_relu(x, negative_slope: @negative_slope)

