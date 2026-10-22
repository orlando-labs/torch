package torch.nn.dropout1d

from torch.nn.dropout import Dropout

export Dropout1d

class Dropout1d < Dropout:
  def init(p: 0.5, seed: 0): _initialize_dropout!(p, seed, :channel1d)
