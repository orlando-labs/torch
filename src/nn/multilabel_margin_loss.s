package torch.nn.multilabel_margin_loss

from torch.nn.module import Module
from torch.nn.functional import multilabel_margin_loss

export MultiLabelMarginLoss

class MultiLabelMarginLoss < Module:
  def init(@reduction: :mean): _initialize_registry!
  def forward(input, target): multilabel_margin_loss(input, target, reduction: @reduction)
