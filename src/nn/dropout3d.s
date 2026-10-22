package torch.nn.dropout3d

from torch.nn.dropout import Dropout

export Dropout3d

class Dropout3d < Dropout:
  def init(p: 0.5, seed: 0): _initialize_dropout!(p, seed, :channel3d)
