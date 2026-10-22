package torch.nn.log_sigmoid

from torch.nn.module import Module
from torch.nn.functional import log_sigmoid

export LogSigmoid

class LogSigmoid < Module:
  def init(): _initialize_registry!
  def forward(x): log_sigmoid(x)

