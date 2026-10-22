package torch.nn.spatial_core

from torch.autograd.context import execution_mode
from torch.tensor.value import Tensor

export spatial, spatial_kernel

native def _spatial(op as Str, inputs, options, mode as Int) from "amt_nn_spatial" blocking

def spatial(value, dimensions):
  result = if Int === value then Array.filled(dimensions, value) else value.array
  raise ValueError("expected #{dimensions} spatial values") if result.count != dimensions
  result

def spatial_kernel(op, inputs, options, pair: false):
  result = _spatial(op, (inputs.map: $it._core), options, execution_mode())
  if pair then (result.map: Tensor($it)).tuple else Tensor(result)
