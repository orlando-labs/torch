package sputniktorch.tensor.bytes

from sputniktorch.tensor.value import Tensor
from sputniktorch.device import device_text
from sputniktorch.autograd.context import execution_mode

export from_bytes

native def _from_bytes(data, offset as Int, shape, device as Str, mode as Int) from "amt_from_bytes"

# Own the copied bytes; the tensor never borrows an Sputnik allocation.
def from_bytes(data, offset: 0, shape: null, device: "cpu"):
  dimensions = if shape == null then null else shape.array
  Tensor(_from_bytes(data, offset, dimensions, device_text(device), execution_mode()))
