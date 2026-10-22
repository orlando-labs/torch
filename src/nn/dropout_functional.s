package torch.nn.dropout_functional

from torch.tensor.value import Tensor
from torch.errors import DTypeError, ShapeError, DeviceError, ONNXError
from torch.autograd.context import execution_mode

export dropout, dropout!, dropout1d, dropout1d!, dropout2d, dropout2d!, dropout3d, dropout3d!
export alpha_dropout, alpha_dropout!, feature_alpha_dropout, feature_alpha_dropout!
export apply_dropout, sample_dropout!, dropout_shape, validate_probability

def validate_probability(p):
  raise ValueError("dropout p must be in 0..1") unless p >= 0 and p <= 1

def dropout_shape(x, kind):
  return x.shape if kind in [:element, :alpha]
  leading = 2
  if kind == :feature_alpha:
    raise ValueError("FeatureAlphaDropout requires rank >= 2") if x.rank < 2
  else:
    dimensions = case! kind.str:
      when "channel1d": 1
      when "channel2d": 2
      when "channel3d": 3
      else: raise ValueError("unknown dropout kind")
    unless x.rank in [dimensions + 1, dimensions + 2]:
      raise ValueError("channel Dropout input has the wrong rank")
    leading = x.rank - dimensions
  Array.of(x.rank) |i|: if i < leading then x.shape[i] else 1

def _check_dtype(x):
  raise DTypeError("stochastic dropout requires floating input") unless x.dtype.floating?

def apply_dropout(x, p, training, kind, mask: null):
  validate_probability(p)
  shape = dropout_shape(x, kind)
  return x if not training or p == 0 or x.numel == 0
  return x * 0 if p == 1
  _check_dtype(x)
  raise ValueError("stochastic dropout requires mask: or a bang call with Generator") if mask == null
  raise TypeError("dropout mask must be a Tensor") unless Tensor === mask
  raise DTypeError("dropout mask must be boolean") unless mask.dtype.bool?
  raise ShapeError("dropout mask must match the noise shape") unless mask.shape == shape
  raise DeviceError("dropout mask must share the input device") unless mask.device.to_str == x.device.to_str
  return x * mask / (1 - p) unless kind in [:alpha, :feature_alpha]
  noise = mask.to(device: x.device, dtype: x.dtype)
  alpha = 1.7580993408473766
  scale = ((alpha * alpha * p + 1) * (1 - p)) ** -0.5
  x * noise * scale + (noise - 1) * (alpha * scale) + alpha * scale * p

def sample_dropout!(x, generator, p, training, kind):
  validate_probability(p)
  shape = dropout_shape(x, kind)
  if not training or p == 0 or p == 1 or x.numel == 0:
    return apply_dropout(x, p, training, kind)
  _check_dtype(x)
  raise ONNXError("stochastic dropout consumes RNG during ONNX export") if execution_mode() // 4 != 0
  mask = generator.rand!(shape, device: x.device).ge(p)
  apply_dropout(x, p, training, kind, mask:)

def dropout(x, p: 0.5, training: true, mask: null): apply_dropout(x, p, training, :element, mask:)
def dropout!(x, generator, p: 0.5, training: true): sample_dropout!(x, generator, p, training, :element)
def dropout1d(x, p: 0.5, training: true, mask: null): apply_dropout(x, p, training, :channel1d, mask:)
def dropout1d!(x, generator, p: 0.5, training: true): sample_dropout!(x, generator, p, training, :channel1d)
def dropout2d(x, p: 0.5, training: true, mask: null): apply_dropout(x, p, training, :channel2d, mask:)
def dropout2d!(x, generator, p: 0.5, training: true): sample_dropout!(x, generator, p, training, :channel2d)
def dropout3d(x, p: 0.5, training: true, mask: null): apply_dropout(x, p, training, :channel3d, mask:)
def dropout3d!(x, generator, p: 0.5, training: true): sample_dropout!(x, generator, p, training, :channel3d)
def alpha_dropout(x, p: 0.5, training: false, mask: null): apply_dropout(x, p, training, :alpha, mask:)
def alpha_dropout!(x, generator, p: 0.5, training: false): sample_dropout!(x, generator, p, training, :alpha)

def feature_alpha_dropout(x, p: 0.5, training: false, mask: null):
  apply_dropout(x, p, training, :feature_alpha, mask:)

def feature_alpha_dropout!(x, generator, p: 0.5, training: false):
  sample_dropout!(x, generator, p, training, :feature_alpha)
