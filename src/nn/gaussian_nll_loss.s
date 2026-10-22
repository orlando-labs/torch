package torch.nn.gaussian_nll_loss

from torch.nn.module import Module
from torch.nn.functional import gaussian_nll_loss

export GaussianNLLLoss

class GaussianNLLLoss < Module:
  def init(@full: false, @eps: 1e-6, @reduction: :mean): _initialize_registry!

  def forward(input, target, var):
    gaussian_nll_loss(input, target, var, full: @full, eps: @eps, reduction: @reduction)
