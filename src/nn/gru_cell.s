package sputniktorch.nn.gru_cell

from sputniktorch.nn.module import Module
from sputniktorch.nn.linear import Linear
from sputniktorch.tensor.factories import zeros

export GRUCell

class GRUCell < Module:
  attr input_projection
  attr hidden_projection

  def init(input_size, @hidden_size, seed: 0, device: "cpu"):
    _initialize_registry!
    @input_projection = Linear(input_size, hidden_size * 3, seed:, device:)
    @hidden_projection = Linear(hidden_size, hidden_size * 3, seed: seed + 1, device:)
    register_module!("input", @input_projection)
    register_module!("hidden", @hidden_projection)

  def forward(x, hidden: null):
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
