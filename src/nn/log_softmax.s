package torch.nn.log_softmax

from torch.nn.module import Module
from torch.nn.functional import log_softmax

export LogSoftmax

class LogSoftmax < Module:
  def init(@dim: -1): _initialize_registry!
  def forward(x): log_softmax(x, dim: @dim)

