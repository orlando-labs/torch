package sputniktorch.nn.nll_loss

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import nll_loss

export NLLLoss

class NLLLoss < Module:
  def init(@weight: null, @reduction: :mean, @ignore_index: -100): _initialize_registry!

  def forward(input, target):
    nll_loss(input, target, weight: @weight,
      reduction: @reduction, ignore_index: @ignore_index)
