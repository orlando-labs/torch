package torch.nn.relu

from torch.nn.module import Module

export ReLU

class ReLU < Module:
  def init(): _initialize_registry!
  def forward(x): x.relu
