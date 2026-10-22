package torch.nn.margin_ranking_loss

from torch.nn.module import Module
from torch.nn.functional import margin_ranking_loss

export MarginRankingLoss

class MarginRankingLoss < Module:
  def init(@margin: 0.0, @reduction: :mean): _initialize_registry!

  def forward(input1, input2, target):
    margin_ranking_loss(input1, input2, target, margin: @margin, reduction: @reduction)
