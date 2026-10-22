package torch.nn.recurrent_core

from torch.tensor.value import Tensor
from torch.errors import ShapeError, DTypeError, DeviceError

export validate_sizes, validate_input, validate_state

def validate_sizes(input_size, hidden_size):
  unless Int === input_size and Int === hidden_size and input_size > 0 and hidden_size > 0:
    raise ValueError("recurrent feature counts must be positive Ints")

def validate_input(x, input_size, ranks):
  raise TypeError("recurrent input must be a Tensor") unless Tensor === x
  raise ShapeError("recurrent input rank or feature count mismatch") unless x.rank in ranks and x.shape[-1] == input_size
  raise DTypeError("recurrent input must be floating") unless x.dtype.floating?

def validate_state(state, x, shape):
  raise TypeError("recurrent state must be a Tensor") unless Tensor === state
  raise ShapeError("recurrent state shape mismatch") unless state.shape == shape
  raise DTypeError("recurrent state dtype mismatch") unless state.dtype == x.dtype
  raise DeviceError("recurrent state device mismatch") unless state.device.to_str == x.device.to_str
