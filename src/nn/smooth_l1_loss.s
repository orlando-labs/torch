package sputniktorch.nn.smooth_l1_loss

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import smooth_l1_loss

export SmoothL1Loss

class SmoothL1Loss < Module:
  def init(@beta: 1.0, @reduction: :mean): _initialize_registry!
  def forward(input, target): smooth_l1_loss(input, target, beta: @beta, reduction: @reduction)

