package torch.nn.soft_margin_loss

from torch.nn.module import Module
from torch.nn.functional import soft_margin_loss

export SoftMarginLoss

class SoftMarginLoss < Module:
  def init(@reduction: :mean): _initialize_registry!
  def forward(input, target): soft_margin_loss(input, target, reduction: @reduction)
