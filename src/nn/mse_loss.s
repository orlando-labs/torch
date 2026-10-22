package torch.nn.mse_loss

from torch.nn.module import Module
from torch.nn.functional import mse_loss

export MSELoss

class MSELoss < Module:
  def init(@reduction: :mean): _initialize_registry!
  def forward(input, target): mse_loss(input, target, reduction: @reduction)

