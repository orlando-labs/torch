package torch.nn.triplet_margin_with_distance_loss

from torch.nn.module import Module
from torch.nn.functional import triplet_margin_with_distance_loss

export TripletMarginWithDistanceLoss

class TripletMarginWithDistanceLoss < Module:
  def init(@distance_function: null, @margin: 1.0, @swap: false, @reduction: :mean):
    _initialize_registry!
    raise ValueError("triplet margin must be positive") unless @margin > 0

  def forward(anchor, positive, negative):
    triplet_margin_with_distance_loss(anchor, positive, negative, distance_function: @distance_function,
      margin: @margin, swap: @swap, reduction: @reduction)
