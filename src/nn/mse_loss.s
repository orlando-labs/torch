package sputniktorch.nn.mse_loss

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import mse_loss

export MSELoss

class MSELoss < Module:
  def init(@reduction: :mean): _initialize_registry!
  def forward(input, target): mse_loss(input, target, reduction: @reduction)

