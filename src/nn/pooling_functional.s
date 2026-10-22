package torch.nn.pooling_functional

from torch.nn.spatial_core import spatial, spatial_kernel

export max_pool1d, max_pool2d, max_pool3d
export adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d
export max_unpool1d, max_unpool2d, max_unpool3d, lp_pool1d, lp_pool2d, lp_pool3d

def _pool(x, dimensions, kernel_size, stride, padding, dilation, ceil_mode, return_indices):
  step = if stride == null then kernel_size else stride
  spatial_kernel("max_pool#{dimensions}d", [x], [spatial(kernel_size, dimensions),
    spatial(step, dimensions), spatial(padding, dimensions), spatial(dilation, dimensions),
    ceil_mode, return_indices], pair: return_indices)

def max_pool1d(x, kernel_size, stride: null, padding: 0, dilation: 1,
    ceil_mode: false, return_indices: false):
  _pool(x, 1, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

def max_pool2d(x, kernel_size, stride: null, padding: 0, dilation: 1,
    ceil_mode: false, return_indices: false):
  _pool(x, 2, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

def max_pool3d(x, kernel_size, stride: null, padding: 0, dilation: 1,
    ceil_mode: false, return_indices: false):
  _pool(x, 3, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

def _adaptive_pool(x, dimensions, output_size, return_indices):
  size = spatial(output_size, dimensions)
  size = Array.of(dimensions) |i|: if size[i] == null then x.shape[-dimensions + i] else size[i]
  spatial_kernel("adaptive_max_pool#{dimensions}d", [x], [size, return_indices], pair: return_indices)

def adaptive_max_pool1d(x, output_size, return_indices: false): _adaptive_pool(x, 1, output_size, return_indices)
def adaptive_max_pool2d(x, output_size, return_indices: false): _adaptive_pool(x, 2, output_size, return_indices)
def adaptive_max_pool3d(x, output_size, return_indices: false): _adaptive_pool(x, 3, output_size, return_indices)

def _unpool(x, indices, dimensions, kernel_size, stride, padding, output_size):
  unless x.rank in [dimensions + 1, dimensions + 2]:
    raise ValueError("MaxUnpool input has the wrong rank")
  kernels = spatial(kernel_size, dimensions)
  strides = spatial(if stride == null then kernel_size else stride, dimensions)
  pads = spatial(padding, dimensions)
  dimensions.times |i|:
    unless kernels[i] > 0 and strides[i] > 0 and pads[i] >= 0:
      raise ValueError("unpool requires positive kernel/stride and nonnegative padding")
  output = Array.of(dimensions) |i|:
    (x.shape[-dimensions + i] - 1) * strides[i] + kernels[i] - 2 * pads[i]
  if output_size != null:
    requested = output_size.array
    requested = requested.last(dimensions) if requested.count == dimensions + 2
    raise ValueError("output_size must contain spatial sizes or [N, C, *spatial]") unless requested.count == dimensions
    dimensions.times |i|:
      unless Int === requested[i] and requested[i] > output[i] - strides[i] and requested[i] < output[i] + strides[i]:
        raise ValueError("output_size is outside the ambiguous pooling range")
    output = requested
  raise ValueError("unpool output sizes must be positive") unless output.all?: $it > 0
  spatial_kernel("max_unpool#{dimensions}d", [x, indices], [output, strides, pads])

def max_unpool1d(x, indices, kernel_size, stride: null, padding: 0, output_size: null):
  _unpool(x, indices, 1, kernel_size, stride, padding, output_size)

def max_unpool2d(x, indices, kernel_size, stride: null, padding: 0, output_size: null):
  _unpool(x, indices, 2, kernel_size, stride, padding, output_size)

def max_unpool3d(x, indices, kernel_size, stride: null, padding: 0, output_size: null):
  _unpool(x, indices, 3, kernel_size, stride, padding, output_size)

def _lp_pool(x, dimensions, norm_type, kernel_size, stride, ceil_mode):
  raise ValueError("norm_type must be positive") unless norm_type > 0
  kernels = spatial(kernel_size, dimensions)
  step = spatial(if stride == null then kernel_size else stride, dimensions)
  volume = kernels.reduce(1) |total, length|: total * length
  average = spatial_kernel("avg_pool#{dimensions}d", [x.pow(norm_type)], [kernels, step,
    Array.filled(dimensions, 0), ceil_mode, true])
  sign = average.gt(0).to(device: x.device, dtype: average.dtype) - average.lt(0).to(device: x.device, dtype: average.dtype)
  (sign * average.abs.relu * volume).pow(1.0 / norm_type)

def lp_pool1d(x, norm_type, kernel_size, stride: null, ceil_mode: false):
  _lp_pool(x, 1, norm_type, kernel_size, stride, ceil_mode)

def lp_pool2d(x, norm_type, kernel_size, stride: null, ceil_mode: false):
  _lp_pool(x, 2, norm_type, kernel_size, stride, ceil_mode)

def lp_pool3d(x, norm_type, kernel_size, stride: null, ceil_mode: false):
  _lp_pool(x, 3, norm_type, kernel_size, stride, ceil_mode)
