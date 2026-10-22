package torch.nn.feature_alpha_dropout

from torch.nn.dropout import Dropout

export FeatureAlphaDropout

class FeatureAlphaDropout < Dropout:
  def init(p: 0.5, seed: 0): _initialize_dropout!(p, seed, :feature_alpha)
