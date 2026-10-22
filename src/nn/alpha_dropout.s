package torch.nn.alpha_dropout

from torch.nn.dropout import Dropout

export AlphaDropout

class AlphaDropout < Dropout:
  def init(p: 0.5, seed: 0): _initialize_dropout!(p, seed, :alpha)
