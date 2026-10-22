package torch.nn.upsample

from torch.nn.module import Module
from torch.nn.interpolation import interpolate

export Upsample

class Upsample < Module:
  def init(size: null, scale_factor: null, mode: :nearest, align_corners: null,
      recompute_scale_factor: null, antialias: false):
    _initialize_upsample!(size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)

  def _initialize_upsample!(size, scale_factor, mode, align_corners, recompute_scale_factor, antialias):
    _initialize_registry!
    @size = size
    @scale_factor = scale_factor
    @mode = mode
    @align_corners = align_corners
    @recompute_scale_factor = recompute_scale_factor
    @antialias = antialias

  def forward(x):
    interpolate(x, size: @size, scale_factor: @scale_factor, mode: @mode,
      align_corners: @align_corners, recompute_scale_factor: @recompute_scale_factor,
      antialias: @antialias)
