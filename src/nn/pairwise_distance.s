package torch.nn.pairwise_distance

from torch.nn.module import Module
from torch.nn.functional import pairwise_distance

export PairwiseDistance

class PairwiseDistance < Module:
  def init(@p: 2.0, @eps: 1e-6, @keepdim: false): _initialize_registry!

  def forward(input1, input2): pairwise_distance(input1, input2, p: @p, eps: @eps, keepdim: @keepdim)
