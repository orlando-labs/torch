package torch.nn.recurrent

from torch.nn.module import Module
from torch.nn.module_list import ModuleList
from torch.nn.rnn_cell import RNNCell
from torch.nn.lstm_cell import LSTMCell
from torch.nn.gru_cell import GRUCell
from torch.nn.functional import stack, cat
from torch.tensor.factories import zeros
from torch.nn.utils.packed_sequence import PackedSequence
from torch.nn.recurrent_core import validate_sizes, validate_input, validate_state
from torch.nn.dropout_functional import validate_probability, sample_dropout!
from torch.random.generator import Generator

export Recurrent

# Hidden state is an explicit input/output value, never shared between calls.
class Recurrent < Module:
  attr cells
  attr input_size
  attr hidden_size
  attr num_layers
  attr bidirectional
  attr batch_first
  attr proj_size

  def _initialize_recurrent!(kind, input_size, hidden_size, num_layers, nonlinearity,
      bias, batch_first, dropout, bidirectional, proj_size, seed, device, dtype):
    _initialize_registry!
    validate_sizes(input_size, hidden_size)
    raise ValueError("num_layers must be a positive Int") unless Int === num_layers and num_layers > 0
    validate_probability(dropout)
    unless Int === proj_size and proj_size >= 0 and proj_size < hidden_size:
      raise ValueError("proj_size must be a nonnegative Int smaller than hidden_size")
    @kind = kind
    @input_size = input_size
    @hidden_size = hidden_size
    @num_layers = num_layers
    @batch_first = batch_first
    @bidirectional = bidirectional
    @directions = if bidirectional then 2 else 1
    @proj_size = proj_size
    @output_size = if proj_size == 0 then hidden_size else proj_size
    @dropout = dropout
    @rng = Generator(seed: seed + num_layers * @directions * 3)
    stride = if proj_size == 0 then 2 else 3
    layers = Array.of(num_layers * @directions) |index|:
      width = if index < @directions then input_size else @output_size * @directions
      cell_seed = seed + index * stride
      case! kind:
        when "rnn": RNNCell(width, hidden_size, nonlinearity:, bias:, seed: cell_seed, device:, dtype:)
        when "lstm": LSTMCell(width, hidden_size, bias:, proj_size:, seed: cell_seed, device:, dtype:)
        when "gru": GRUCell(width, hidden_size, bias:, seed: cell_seed, device:, dtype:)
        else: raise ValueError("unknown recurrent kind")
    @cells = ModuleList(layers)

  def stateful?(): training? and @num_layers > 1 and @dropout > 0 and @dropout < 1

  def rng_state(): @rng.state

  def load_rng_state!(state):
    @rng.load_state!(state)
    self

  def _prepare(x, state):
    validate_input(x, @input_size, [2, 3])
    unbatched = x.rank == 2
    batch = if unbatched then 1 else if @batch_first then x.shape[0] else x.shape[1]
    time = if unbatched or not @batch_first then x.shape[0] else x.shape[1]
    raise ValueError("recurrent sequence length must be positive") if time == 0
    if state != null:
      leading = if unbatched then [@num_layers * @directions] else [@num_layers * @directions, batch]
      if @kind == "lstm":
        raise TypeError("LSTM state must be a (hidden, cell) pair") unless (Tuple === state or Array === state) and state.count == 2
        validate_state(state[0], x, [*leading, @output_size])
        validate_state(state[1], x, [*leading, @hidden_size])
      else:
        validate_state(state, x, [*leading, @output_size])
    if unbatched:
      x = x.unsqueeze(0)
      if state != null:
        state = if @kind == "lstm" then (state[0].unsqueeze(1), state[1].unsqueeze(1)) else state.unsqueeze(1)
    elif not @batch_first:
      x = x.transpose(0, 1)
    return (x, state, unbatched)

  def _forward(x, state, mutate):
    raise ValueError("recurrent dropout consumes RNG; use call! or eval!") if stateful? and not mutate
    return _forward_packed(x, state) if PackedSequence === x
    prepared = _prepare(x, state)
    (x, state, unbatched) = prepared

    states = []
    @num_layers.times |layer|:
      directions = Array.of(@directions) |direction|:
        index = layer * @directions + direction
        cell = @cells[index]
        current = if state == null:
          null
        else if @kind == "lstm":
          (state[0][index], state[1][index])
        else:
          state[index]
        outputs = Array.of(x.shape[1]) |step|:
          time = if direction == 0 then step else x.shape[1] - step - 1
          input = x.select(1, time)
          current = if @kind == "lstm" then cell(input, state: current) else cell(input, hidden: current)
          if @kind == "lstm" then current[0] else current
        states.push!(current)
        outputs = outputs.reversed if direction == 1
        stack(outputs, dim: 1)
      x = if @directions == 1 then directions[0] else cat(directions, dim: -1)
      if layer < @num_layers - 1:
        x = sample_dropout!(x, @rng, @dropout, training?, :element)

    x = if unbatched then x.squeeze(dim: 0) else if @batch_first then x else x.transpose(0, 1)
    if @kind == "lstm":
      hidden = states.map |entry|: entry[0]
      memory = states.map |entry|: entry[1]
      h = stack(hidden)
      c = stack(memory)
      h = h.squeeze(dim: 1) if unbatched
      c = c.squeeze(dim: 1) if unbatched
      return (x, (h, c))
    else:
      h = stack(states)
      h = h.squeeze(dim: 1) if unbatched
      return (x, h)

  def _packed_state(sequence, state, batch):
    x = sequence.data
    validate_input(x, @input_size, [2])
    shape = [@num_layers * @directions, batch]
    if state != null:
      if @kind == "lstm":
        raise TypeError("LSTM state must be a (hidden, cell) pair") unless (Tuple === state or Array === state) and state.count == 2
        validate_state(state[0], x, [*shape, @output_size])
        validate_state(state[1], x, [*shape, @hidden_size])
      else:
        validate_state(state, x, [*shape, @output_size])
    else:
      hidden = zeros([*shape, @output_size], dtype: x.dtype, device: x.device)
      state = if @kind == "lstm" then (hidden, zeros([*shape, @hidden_size], dtype: x.dtype, device: x.device)) else hidden
    if sequence.sorted_indices != null:
      indices = sequence.sorted_indices
      state = if @kind == "lstm" then (state[0].index_select(1, indices), state[1].index_select(1, indices)) else state.index_select(1, indices)
    state

  def _prefix(state, count):
    if @kind == "lstm" then (state[0].narrow(0, 0, count), state[1].narrow(0, 0, count)) else state.narrow(0, 0, count)

  def _replace_prefix(previous, current, count, batch):
    return current if count == batch
    if @kind == "lstm":
      return (cat([current[0], previous[0].narrow(0, count, batch - count)]),
        cat([current[1], previous[1].narrow(0, count, batch - count)]))
    else:
      cat([current, previous.narrow(0, count, batch - count)])

  def _forward_packed(sequence, state):
    sizes = sequence.batch_sizes.to_a
    batch = sizes[0]
    state = _packed_state(sequence, state, batch)
    offset = 0
    offsets = sizes.map |size|:
      start = offset
      offset = offset + size
      start
    x = sequence.data
    states = []
    @num_layers.times |layer|:
      directions = Array.of(@directions) |direction|:
        index = layer * @directions + direction
        cell = @cells[index]
        current = if @kind == "lstm" then (state[0][index], state[1][index]) else state[index]
        outputs = Array.of(sizes.count) |step|:
          time = if direction == 0 then step else sizes.count - step - 1
          count = sizes[time]
          input = x.narrow(0, offsets[time], count)
          active = _prefix(current, count)
          next_state = if @kind == "lstm" then cell(input, state: active) else cell(input, hidden: active)
          # Completed forward rows retain their final state; newly active reverse
          # rows still hold their initial state until their first real timestep.
          current = _replace_prefix(current, next_state, count, batch)
          if @kind == "lstm" then next_state[0] else next_state
        states.push!(current)
        outputs = outputs.reversed if direction == 1
        cat(outputs)
      x = if @directions == 1 then directions[0] else cat(directions, dim: -1)
      if layer < @num_layers - 1:
        x = sample_dropout!(x, @rng, @dropout, training?, :element)
    output = PackedSequence(x, sequence.batch_sizes, sorted_indices: sequence.sorted_indices,
      unsorted_indices: sequence.unsorted_indices)
    hidden = stack(states.map: if @kind == "lstm" then $it[0] else $it)
    hidden = hidden.index_select(1, sequence.unsorted_indices) if sequence.unsorted_indices != null
    if @kind == "lstm":
      memory = stack(states.map: $it[1])
      memory = memory.index_select(1, sequence.unsorted_indices) if sequence.unsorted_indices != null
      return (output, (hidden, memory))
    else:
      return (output, hidden)

  def forward(x, state: null): _forward(x, state, false)
  def forward!(x, state: null): _forward(x, state, true)
