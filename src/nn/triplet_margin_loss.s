package torch.nn.triplet_margin_loss

from torch.nn.module import Module
from torch.nn.functional import triplet_margin_loss

export TripletMarginLoss

class TripletMarginLoss < Module:
  def init(@margin: 1.0, @p: 2.0, @eps: 1e-6, @swap: false, @reduction: :mean):
    _initialize_registry!
    raise ValueError("triplet margin must be positive") unless @margin > 0

  def forward(anchor, positive, negative):
    triplet_margin_loss(anchor, positive, negative, margin: @margin, p: @p,
      eps: @eps, swap: @swap, reduction: @reduction)
