package sputniktorch.nn.l1_loss

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import l1_loss

export L1Loss

class L1Loss < Module:
  def init(@reduction: :mean): _initialize_registry!
  def forward(input, target): l1_loss(input, target, reduction: @reduction)

