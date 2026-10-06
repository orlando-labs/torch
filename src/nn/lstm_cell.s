package sputniktorch.nn.lstm_cell

from sputniktorch.nn.module import Module
from sputniktorch.nn.linear import Linear
from sputniktorch.tensor.factories import zeros

export LSTMCell

class LSTMCell < Module:
  attr input_projection
  attr hidden_projection

  def init(input_size, @hidden_size, seed: 0, device: "cpu"):
    _initialize_registry!
    @input_projection = Linear(input_size, hidden_size * 4, seed:, device:)
    @hidden_projection = Linear(hidden_size, hidden_size * 4, seed: seed + 1, device:)
    register_module!("input", @input_projection)
    register_module!("hidden", @hidden_projection)

  def forward(x, state: null):
    (hidden, cell) = if state == null:
      shape = [*x.shape[0...-1], @hidden_size]
      (zeros(shape, dtype: x.dtype, device: x.device), zeros(shape, dtype: x.dtype, device: x.device))
    else:
      state
    gates = @input_projection(x) + @hidden_projection(hidden)
    [input_gate, forget_gate, candidate, output_gate] = gates.chunk(4, dim: -1)
    next_cell = forget_gate.sigmoid * cell + input_gate.sigmoid * candidate.tanh
    (output_gate.sigmoid * next_cell.tanh, next_cell)
