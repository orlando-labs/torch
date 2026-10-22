package torch.nn.lstm_cell

from torch.nn.module import Module
from torch.nn.linear import Linear
from torch.tensor.factories import zeros
from torch.nn.recurrent_core import validate_sizes, validate_input, validate_state

export LSTMCell

class LSTMCell < Module:
  attr input_projection
  attr hidden_projection
  attr output_projection

  def init(@input_size, @hidden_size, bias: true, @proj_size: 0, seed: 0,
      device: "cpu", dtype: "float32"):
    _initialize_registry!
    validate_sizes(input_size, hidden_size)
    unless Int === proj_size and proj_size >= 0 and proj_size < hidden_size:
      raise ValueError("proj_size must be a nonnegative Int smaller than hidden_size")
    @output_size = if proj_size == 0 then hidden_size else proj_size
    @input_projection = Linear(input_size, hidden_size * 4, bias:, seed:, device:, dtype:)
    @hidden_projection = Linear(@output_size, hidden_size * 4, bias:, seed: seed + 1, device:, dtype:)
    @output_projection = if proj_size == 0 then null else Linear(hidden_size, proj_size, bias: false, seed: seed + 2, device:, dtype:)
    register_module!("input", @input_projection)
    register_module!("hidden", @hidden_projection)
    register_module!("projection", @output_projection) if @output_projection != null

  def forward(x, state: null):
    validate_input(x, @input_size, [1, 2])
    if state != null:
      raise TypeError("LSTM state must be a (hidden, cell) pair") unless (Tuple === state or Array === state) and state.count == 2
      validate_state(state[0], x, [*x.shape[0...-1], @output_size])
      validate_state(state[1], x, [*x.shape[0...-1], @hidden_size])
    prepared_state = if state == null:
      (zeros([*x.shape[0...-1], @output_size], dtype: x.dtype, device: x.device),
        zeros([*x.shape[0...-1], @hidden_size], dtype: x.dtype, device: x.device))
    else:
      state
    (hidden, cell) = prepared_state
    gates = @input_projection(x) + @hidden_projection(hidden)
    [input_gate, forget_gate, candidate, output_gate] = gates.chunk(4, dim: -1)
    next_cell = forget_gate.sigmoid * cell + input_gate.sigmoid * candidate.tanh
    next_hidden = output_gate.sigmoid * next_cell.tanh
    next_hidden = @output_projection(next_hidden) if @output_projection != null
    return (next_hidden, next_cell)
