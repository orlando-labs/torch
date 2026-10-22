package torch.nn.upsampling_bilinear2d

from torch.nn.upsample import Upsample

export UpsamplingBilinear2d

class UpsamplingBilinear2d < Upsample:
  def init(size: null, scale_factor: null):
    _initialize_upsample!(size, scale_factor, :bilinear, true, null, false)
