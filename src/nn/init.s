package sputniktorch.nn.init

from sputniktorch.autograd import no_grad
from sputniktorch.random.generator import Generator

export zeros!, ones!, constant!, uniform!, normal!, xavier_uniform!, xavier_normal!, kaiming_uniform!, kaiming_normal!

def constant!(tensor, value): no_grad(): tensor.fill!(value)
def zeros!(tensor): constant!(tensor, 0)
def ones!(tensor): constant!(tensor, 1)
def _write!(tensor, value): no_grad(): tensor.copy_from!(value)

def uniform!(tensor, low: 0.0, high: 1.0, seed: 0):
  raise ValueError("uniform bounds are reversed") if high < low
  noise = Generator(seed:).rand!(tensor.shape, dtype: tensor.dtype, device: tensor.device)
  _write!(tensor, noise * (high - low) + low)

def normal!(tensor, mean: 0.0, std: 1.0, seed: 0):
  raise ValueError("normal std must be nonnegative") if std < 0
  noise = Generator(seed:).randn!(tensor.shape, dtype: tensor.dtype, device: tensor.device)
  _write!(tensor, noise * std + mean)

def _fans(tensor):
  raise ValueError("fan initialization needs at least two axes") if tensor.rank < 2
  spatial = tensor.shape.drop(2).product
  (tensor.shape[1] * spatial, tensor.shape[0] * spatial)

def xavier_uniform!(tensor, gain: 1.0, seed: 0):
  (fan_in, fan_out) = _fans(tensor)
  bound = gain * (6.0 / (fan_in + fan_out)) ** 0.5
  uniform!(tensor, low: -bound, high: bound, seed:)

def xavier_normal!(tensor, gain: 1.0, seed: 0):
  (fan_in, fan_out) = _fans(tensor)
  normal!(tensor, std: gain * (2.0 / (fan_in + fan_out)) ** 0.5, seed:)

def _fan(tensor, mode):
  (fan_in, fan_out) = _fans(tensor)
  case! mode.str:
    when "fan_in": fan_in
    when "fan_out": fan_out
    else: raise ValueError("mode must be :fan_in or :fan_out")

def kaiming_uniform!(tensor, negative_slope: 0.0, mode: :fan_in, seed: 0):
  bound = (6.0 / ((1 + negative_slope ** 2) * _fan(tensor, mode))) ** 0.5
  uniform!(tensor, low: -bound, high: bound, seed:)

def kaiming_normal!(tensor, negative_slope: 0.0, mode: :fan_in, seed: 0):
  std = (2.0 / ((1 + negative_slope ** 2) * _fan(tensor, mode))) ** 0.5
  normal!(tensor, std:, seed:)
