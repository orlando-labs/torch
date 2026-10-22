package torch.nn.huber_loss

from torch.nn.module import Module
from torch.nn.functional import huber_loss

export HuberLoss

class HuberLoss < Module:
  def init(@delta: 1.0, @reduction: :mean): _initialize_registry!
  def forward(input, target): huber_loss(input, target, delta: @delta, reduction: @reduction)

