package torch.nn.local_response_norm

from torch.nn.module import Module
from torch.nn.functional import local_response_norm

export LocalResponseNorm

class LocalResponseNorm < Module:
  def init(@size, @alpha: 1e-4, @beta: 0.75, @k: 1.0):
    _initialize_registry!
    raise ValueError("size must be a positive Int") unless Int === @size and @size > 0

  def forward(x): local_response_norm(x, @size, alpha: @alpha, beta: @beta, k: @k)
