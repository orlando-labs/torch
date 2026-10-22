package torch.nn.l1_loss

from torch.nn.module import Module
from torch.nn.functional import l1_loss

export L1Loss

class L1Loss < Module:
  def init(@reduction: :mean): _initialize_registry!
  def forward(input, target): l1_loss(input, target, reduction: @reduction)

