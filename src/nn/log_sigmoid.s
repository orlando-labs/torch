package sputniktorch.nn.log_sigmoid

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import log_sigmoid

export LogSigmoid

class LogSigmoid < Module:
  def init(): _initialize_registry!
  def forward(x): log_sigmoid(x)

