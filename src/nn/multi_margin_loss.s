package torch.nn.multi_margin_loss

from torch.nn.module import Module
from torch.nn.functional import multi_margin_loss

export MultiMarginLoss

class MultiMarginLoss < Module:
  attr weight

  def init(@p: 1, @margin: 1.0, @weight: null, @reduction: :mean):
    _initialize_registry!
    raise ValueError("multi-margin p must be 1 or 2") unless @p in [1, 2]
    register_buffer!("weight", @weight) if @weight != null

  def forward(input, target):
    multi_margin_loss(input, target, p: @p, margin: @margin, weight: @weight, reduction: @reduction)
