package sputniktorch.nn.rnn_cell

from sputniktorch.nn.module import Module
from sputniktorch.nn.linear import Linear
from sputniktorch.tensor.factories import zeros

export RNNCell

class RNNCell < Module:
  attr input_projection
  attr hidden_projection

  def init(input_size, @hidden_size, @nonlinearity: :tanh, seed: 0, device: "cpu"):
    _initialize_registry!
    raise ValueError("expected :tanh or :relu") unless (nonlinearity in [:tanh, :relu])
    @input_projection = Linear(input_size, hidden_size, seed:, device:)
    @hidden_projection = Linear(hidden_size, hidden_size, seed: seed + 1, device:)
    register_module!("input", @input_projection)
    register_module!("hidden", @hidden_projection)

  def forward(x, hidden: null):
    state = if hidden == null:
      zeros([*x.shape[0...-1], @hidden_size], dtype: x.dtype, device: x.device)
    else:
      hidden
    output = @input_projection(x) + @hidden_projection(state)
    case! @nonlinearity.str:
      when "tanh": output.tanh
      when "relu": output.relu
