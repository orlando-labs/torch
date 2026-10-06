package sputniktorch.nn.log_softmax

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import log_softmax

export LogSoftmax

class LogSoftmax < Module:
  def init(@dim: -1): _initialize_registry!
  def forward(x): log_softmax(x, dim: @dim)

