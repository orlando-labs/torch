package torch.nn.kl_div_loss

from torch.nn.module import Module
from torch.nn.functional import kl_div

export KLDivLoss

class KLDivLoss < Module:
  def init(@log_target: false, @reduction: :mean): _initialize_registry!
  def forward(input, target): kl_div(input, target, log_target: @log_target, reduction: @reduction)

