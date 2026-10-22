package torch.nn.bce_loss

from torch.nn.module import Module
from torch.nn.functional import binary_cross_entropy

export BCELoss

class BCELoss < Module:
  def init(@weight: null, @reduction: :mean): _initialize_registry!

  def forward(input, target):
    binary_cross_entropy(input, target, weight: @weight, reduction: @reduction)
