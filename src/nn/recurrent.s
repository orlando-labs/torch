package sputniktorch.nn.recurrent

from sputniktorch.nn.module import Module
from sputniktorch.nn.module_list import ModuleList
from sputniktorch.nn.rnn_cell import RNNCell
from sputniktorch.nn.lstm_cell import LSTMCell
from sputniktorch.nn.gru_cell import GRUCell
from sputniktorch.nn.functional import stack

export Recurrent

# Batch-first sequences. State is an explicit input/output value and never
# lives inside the module, so independent sequences cannot share hidden state.
class Recurrent < Module:
  attr cells

  def _initialize_recurrent!(kind, input_size, hidden_size, num_layers, nonlinearity, seed, device):
    _initialize_registry!
    raise ValueError("num_layers must be positive") if num_layers <= 0
    @kind = kind
    layers = Array.of(num_layers) |index|:
      width = if index == 0 then input_size else hidden_size
      case! kind:
        when "rnn": RNNCell(width, hidden_size, nonlinearity:, seed: seed + index * 2, device:)
        when "lstm": LSTMCell(width, hidden_size, seed: seed + index * 2, device:)
        when "gru": GRUCell(width, hidden_size, seed: seed + index * 2, device:)
        else: raise ValueError("unknown recurrent kind")
    @cells = ModuleList(layers)

  def forward(x, state: null):
    valid_sequence = x.rank == 3 and x.shape[1] > 0
    raise ValueError("expected a nonempty [batch, time, features] sequence") unless valid_sequence

    states = []
    @cells.count.times |index|:
      cell = @cells[index]
      current = if state == null:
        null
      else if @kind == "lstm":
        (state[0][index], state[1][index])
      else:
        state[index]
      outputs = Array.of(x.shape[1]) |time|:
        input = x.select(1, time)
        current = if @kind == "lstm":
          cell(input, state: current)
        else:
          cell(input, hidden: current)
        if @kind == "lstm" then current[0] else current
      states.push!(current)
      x = stack(outputs, dim: 1)

    if @kind == "lstm":
      hidden = states.map |entry|: entry[0]
      memory = states.map |entry|: entry[1]
      (x, (stack(hidden), stack(memory)))
    else:
      (x, stack(states))
