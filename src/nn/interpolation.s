package torch.nn.interpolation

from torch.nn.spatial_core import spatial, spatial_kernel

export interpolate

def interpolate(input, size: null, scale_factor: null, mode: :nearest,
    align_corners: null, recompute_scale_factor: null, antialias: false):
  dimensions = input.rank - 2
  raise ValueError("interpolate requires rank 3, 4 or 5") unless dimensions in [1, 2, 3]
  if (size == null) == (scale_factor == null):
    raise ValueError("provide exactly one of size: and scale_factor:")
  name = mode.str
  raise ValueError("unknown interpolation mode") unless name in ["nearest", "nearest-exact",
    "linear", "bilinear", "bicubic", "trilinear", "area", "lanczos"]
  required = {linear: 1, bilinear: 2, bicubic: 2, trilinear: 3, lanczos: 2}
  if required.has_key?(name) and dimensions != required[name]:
    raise ValueError("interpolation mode does not match the spatial rank")
  if name in ["nearest", "nearest-exact", "area"] and align_corners != null:
    raise ValueError("align_corners applies only to linear, bilinear, bicubic and trilinear")
  if antialias and not (dimensions == 2 and name in ["bilinear", "bicubic", "lanczos"]):
    raise ValueError("antialias requires 2D bilinear, bicubic or lanczos")
  raise ValueError("lanczos requires antialias: true") if name == "lanczos" and not antialias
  if size != null and recompute_scale_factor == true:
    raise ValueError("recompute_scale_factor requires scale_factor:")
  if size != null and not (Int === size or Array === size or Tuple === size):
    raise ValueError("size must be an Int or an Array/Tuple of Ints")
  if scale_factor != null and not (Int === scale_factor or Float === scale_factor or Array === scale_factor or Tuple === scale_factor):
    raise ValueError("scale_factor must be numeric or an Array/Tuple of numbers")
  output = if size == null then null else spatial(size, dimensions)
  if output != null:
    output.each |length|:
      raise ValueError("output size must contain positive Ints") unless Int === length and length > 0
  scales = if scale_factor == null:
    null
  elif Int === scale_factor or Float === scale_factor:
    Array.filled(dimensions, scale_factor)
  else:
    scale_factor.array
  if scales != null:
    raise ValueError("scale_factor must match the spatial rank") unless scales.count == dimensions
  spatial_kernel("interpolate", [input], [output, scales, name, align_corners,
    recompute_scale_factor, antialias])
