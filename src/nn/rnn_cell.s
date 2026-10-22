package torch.nn.rnn_cell

from torch.nn.module import Module
from torch.nn.linear import Linear
from torch.tensor.factories import zeros
from torch.nn.recurrent_core import validate_sizes, validate_input, validate_state

export RNNCell

class RNNCell < Module:
  attr input_projection
  attr hidden_projection

  def init(@input_size, @hidden_size, nonlinearity: :tanh, bias: true, seed: 0,
      device: "cpu", dtype: "float32"):
    _initialize_registry!
    validate_sizes(input_size, hidden_size)
    @nonlinearity = nonlinearity.str
    raise ValueError("expected :tanh or :relu") unless @nonlinearity in ["tanh", "relu"]
    @input_projection = Linear(input_size, hidden_size, bias:, seed:, device:, dtype:)
    @hidden_projection = Linear(hidden_size, hidden_size, bias:, seed: seed + 1, device:, dtype:)
    register_module!("input", @input_projection)
    register_module!("hidden", @hidden_projection)

  def forward(x, hidden: null):
    validate_input(x, @input_size, [1, 2])
    validate_state(hidden, x, [*x.shape[0...-1], @hidden_size]) if hidden != null
    state = if hidden == null:
      zeros([*x.shape[0...-1], @hidden_size], dtype: x.dtype, device: x.device)
    else:
      hidden
    output = @input_projection(x) + @hidden_projection(state)
    case! @nonlinearity.str:
      when "tanh": output.tanh
      when "relu": output.relu
