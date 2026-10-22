package torch.nn.upsampling_nearest2d

from torch.nn.upsample import Upsample

export UpsamplingNearest2d

class UpsamplingNearest2d < Upsample:
  def init(size: null, scale_factor: null):
    _initialize_upsample!(size, scale_factor, :nearest, null, null, false)
