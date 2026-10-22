package torch.nn.poisson_nll_loss

from torch.nn.module import Module
from torch.nn.functional import poisson_nll_loss

export PoissonNLLLoss

class PoissonNLLLoss < Module:
  def init(@log_input: true, @full: false, @eps: 1e-8, @reduction: :mean): _initialize_registry!

  def forward(input, target):
    poisson_nll_loss(input, target, log_input: @log_input, full: @full, eps: @eps, reduction: @reduction)
