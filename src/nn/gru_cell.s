package torch.nn.gru_cell

from torch.nn.module import Module
from torch.nn.linear import Linear
from torch.tensor.factories import zeros
from torch.nn.recurrent_core import validate_sizes, validate_input, validate_state

export GRUCell

class GRUCell < Module:
  attr input_projection
  attr hidden_projection

  def init(@input_size, @hidden_size, bias: true, seed: 0, device: "cpu", dtype: "float32"):
    _initialize_registry!
    validate_sizes(input_size, hidden_size)
    @input_projection = Linear(input_size, hidden_size * 3, bias:, seed:, device:, dtype:)
    @hidden_projection = Linear(hidden_size, hidden_size * 3, bias:, seed: seed + 1, device:, dtype:)
    register_module!("input", @input_projection)
    register_module!("hidden", @hidden_projection)

  def forward(x, hidden: null):
    validate_input(x, @input_size, [1, 2])
    validate_state(hidden, x, [*x.shape[0...-1], @hidden_size]) if hidden != null
    state = if hidden == null:
      zeros([*x.shape[0...-1], @hidden_size], dtype: x.dtype, device: x.device)
    else:
      hidden
    [input_reset, input_update, input_candidate] = @input_projection(x).chunk(3, dim: -1)
    [hidden_reset, hidden_update, hidden_candidate] = @hidden_projection(state).chunk(3, dim: -1)
    reset = (input_reset + hidden_reset).sigmoid
    update = (input_update + hidden_update).sigmoid
    candidate = (input_candidate + reset * hidden_candidate).tanh
    candidate + update * (state - candidate)
