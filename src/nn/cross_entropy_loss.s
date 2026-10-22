package torch.nn.cross_entropy_loss

from torch.nn.module import Module
from torch.nn.functional import cross_entropy

export CrossEntropyLoss

class CrossEntropyLoss < Module:
  def init(@weight: null, @reduction: :mean, @ignore_index: -100, @label_smoothing: 0.0):
    _initialize_registry!

  def forward(input, target):
    cross_entropy(input, target, weight: @weight, reduction: @reduction,
      ignore_index: @ignore_index, label_smoothing: @label_smoothing)
