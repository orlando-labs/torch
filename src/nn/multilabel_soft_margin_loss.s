package torch.nn.multilabel_soft_margin_loss

from torch.nn.module import Module
from torch.nn.functional import multilabel_soft_margin_loss

export MultiLabelSoftMarginLoss

class MultiLabelSoftMarginLoss < Module:
  attr weight

  def init(@weight: null, @reduction: :mean):
    _initialize_registry!
    register_buffer!("weight", @weight) if @weight != null

  def forward(input, target):
    multilabel_soft_margin_loss(input, target, weight: @weight, reduction: @reduction)
