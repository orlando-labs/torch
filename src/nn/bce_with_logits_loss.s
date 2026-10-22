package torch.nn.bce_with_logits_loss

from torch.nn.module import Module
from torch.nn.functional import binary_cross_entropy_with_logits

export BCEWithLogitsLoss

class BCEWithLogitsLoss < Module:
  def init(@weight: null, @pos_weight: null, @reduction: :mean): _initialize_registry!

  def forward(input, target):
    binary_cross_entropy_with_logits(input, target,
      weight: @weight, pos_weight: @pos_weight, reduction: @reduction)
