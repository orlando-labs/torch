package torch.nn.dropout2d

from torch.nn.dropout import Dropout

export Dropout2d

class Dropout2d < Dropout:
  def init(p: 0.5, seed: 0): _initialize_dropout!(p, seed, :channel2d)
