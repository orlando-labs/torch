package torch.recurrent_test

from torch import tensor, randn, zeros, ones
from torch.tensor.value import Tensor
from torch.nn import RNN, LSTM, GRU, RNNCell, LSTMCell, GRUCell, Sequential, Linear
from torch.autograd import no_grad, inference_mode
from torch.checks import check

export main, run

native def _reference(kind, input, parameters, state, bias, layers, bidirectional, batch_first) from "amt_recurrent_reference" blocking

def close(actual, expected, message):
  actual = actual.to(device: "cpu")
  expected = expected.to(device: "cpu")
  check(actual.shape == expected.shape and (actual - expected).abs.sum.item < 8e-4, message)

def rejected(&block):
  failed = false
  try:
    block()
  rescue ValueError:
    failed = true
  rescue TypeError:
    failed = true
  rescue Torch.Errors.Error:
    failed = true
  check(failed, "invalid recurrent call must raise")

def weights(model):
  result = []
  model.cells.each |cell|:
    result.push!(cell.input_projection.weight)
    result.push!(cell.hidden_projection.weight)
    if cell.input_projection.bias != null:
      result.push!(cell.input_projection.bias)
      result.push!(cell.hidden_projection.bias)
    if LSTMCell === cell and cell.output_projection != null:
      result.push!(cell.output_projection.weight)
  result

def make_model(kind, device, dtype, bias, batch_first, bidirectional, layers, projection = 0, dropout = 0):
  case! kind:
    when "lstm": LSTM(3, 4, num_layers: layers, bias:, batch_first:, bidirectional:,
      proj_size: projection, dropout:, seed: 13, device:, dtype:)
    when "gru": GRU(3, 4, num_layers: layers, bias:, batch_first:, bidirectional:,
      dropout:, seed: 13, device:, dtype:)
    else: RNN(3, 4, num_layers: layers, nonlinearity: kind, bias:, batch_first:,
      bidirectional:, dropout:, seed: 13, device:, dtype:)

def compare_reference(kind, device, dtype, bias, batch_first, bidirectional, layers, projection = 0):
  model = make_model(kind, device, dtype, bias, batch_first, bidirectional, layers, projection)
  # Nonzero biases expose the GRU reset-gate placement as well as all LSTM gates.
  no_grad():
    model.cells.each |cell|:
      cell.input_projection.bias.fill!(0.07) if bias
      cell.hidden_projection.bias.fill!(-0.03) if bias
  shape = if batch_first then [2, 3, 3] else [3, 2, 3]
  x = randn(shape, seed: 9, device:, dtype:) * 0.2
  x = x.detach.requires_grad!(true)
  width = if projection == 0 then 4 else projection
  directions = if bidirectional then 2 else 1
  h = randn([layers * directions, 2, width], seed: 10, device:, dtype:).requires_grad!(true)
  c = randn([layers * directions, 2, 4], seed: 11, device:, dtype:).requires_grad!(true)
  state = if kind == "lstm" then (h, c) else h
  parameters = weights(model)
  states = if kind == "lstm" then [h._core, c._core] else [h._core]
  handles = parameters.map: $it._core
  expected = _reference(kind, x._core, handles, states, bias, layers, bidirectional, batch_first)
  (output, final) = model(x, state:)
  hidden = if kind == "lstm" then final[0] else final
  close(output, Tensor(expected[0]), "ATen recurrent output " + kind)
  close(hidden, Tensor(expected[1]), "ATen recurrent final hidden " + kind)
  loss = output.square.sum + hidden.square.sum
  if kind == "lstm":
    close(final[1], Tensor(expected[2]), "ATen recurrent final memory")
    loss = loss + final[1].square.sum
  loss.backward!
  close(x.grad, Tensor(expected[3]), "ATen recurrent input gradient")
  close(h.grad, Tensor(expected[4]), "ATen recurrent initial hidden gradient")
  close(c.grad, Tensor(expected[5]), "ATen recurrent initial memory gradient") if kind == "lstm"
  parameters.each_with_index |parameter, index|:
    close(parameter.grad, Tensor(expected[6][index]), "ATen recurrent parameter gradient")
    check(parameter.leaf? and parameter.grad.device.type == device, "recurrent leaf and gradient backend")
  check(output.device.type == device and hidden.device.type == device, "recurrent output backend")

def run(device):
  ["tanh", "relu", "gru", "lstm"].each |kind|:
    compare_reference(kind, device, "float32", true, true, true, 2)
    compare_reference(kind, device, "float32", false, false, false, 2)
  compare_reference("lstm", device, "float32", true, false, true, 2, 2)
  compare_reference("lstm", device, "float32", false, true, false, 1, 2)
  compare_reference("gru", device, "float64", true, false, true, 2) if device == "cpu"

  x = randn([5, 3], seed: 1, device:)
  ["tanh", "gru", "lstm"].each |kind|:
    model = make_model(kind, device, "float32", false, false, true, 2)
    (output, final) = model(x)
    (batched, batched_final) = model(x.unsqueeze(1))
    close(output, batched.squeeze(dim: 1), "unbatched ignores batch-first")
    if kind == "lstm":
      check(final[0].shape == [4, 4] and final[1].shape == [4, 4], "unbatched LSTM state layout")
      close(final[0], batched_final[0].squeeze(dim: 1), "unbatched hidden")
      close(final[1], batched_final[1].squeeze(dim: 1), "unbatched memory")
    else:
      check(final.shape == [4, 4], "unbatched recurrent state layout")
      close(final, batched_final.squeeze(dim: 1), "unbatched hidden")
    (continued, final_again) = model(x, state: final)
    check(continued.shape == [5, 8], "unbatched explicit state continuation")
    restored = make_model(kind, device, "float32", false, false, true, 2)
    restored.load_state_dict!(model.state_dict)
    close(restored(x)[0], output, "bidirectional state dict round trip")
    # The backward direction's final state corresponds to original time zero.
    last_hidden = if kind == "lstm" then final[0] else final
    close(last_hidden[2], output[4].narrow(0, 0, 4), "forward final-state ordering")
    close(last_hidden[3], output[0].narrow(0, 4, 4), "reverse final-state ordering")

  sequence = randn([2, 4, 3], seed: 2, device:)
  noisy = GRU(3, 4, num_layers: 2, bidirectional: true, dropout: 0.5, seed: 4, device:)
  before = noisy.rng_state
  rejected(): noisy(sequence)
  close(noisy.rng_state, before, "non-bang does not consume RNG")
  first = noisy.call!(sequence)[0]
  check((noisy.rng_state - before).abs.sum.item > 0, "inter-layer dropout consumes RNG")
  noisy.load_rng_state!(before)
  close(noisy.call!(sequence)[0], first, "recurrent RNG replay")
  bad_state = zeros([4, 2, 5], device:)
  before_invalid = noisy.rng_state
  rejected(): noisy.call!(sequence, state: bad_state)
  close(noisy.rng_state, before_invalid, "invalid state does not consume RNG")
  container = Sequential([Linear(3, 3, device:), noisy])
  rejected(): container(sequence)
  close(noisy.rng_state, before_invalid, "Sequential rejects before stochastic child")
  noisy.eval!
  clean = GRU(3, 4, num_layers: 2, bidirectional: true, seed: 4, device:)
  close(noisy(sequence)[0], clean(sequence)[0], "eval disables inter-layer dropout")
  close(noisy.rng_state, before_invalid, "eval leaves RNG unchanged")
  singleton = GRU(3, 4, dropout: 0.9, device:)
  check(not singleton.stateful?, "single layer has no dropout")
  singleton_rng = singleton.rng_state
  singleton(sequence)
  close(singleton.rng_state, singleton_rng, "single layer leaves RNG unchanged")
  all_dropped = RNN(3, 4, num_layers: 2, dropout: 1.0, bias: false, device:)
  close(all_dropped(sequence)[0], zeros([2, 4, 4], device:), "dropout p1 only between layers")
  final_unmasked = RNN(3, 4, num_layers: 2, dropout: 1.0, device:)
  no_grad():
    final_unmasked.parameters.each: $it.zero!
    final_unmasked.cells[1].input_projection.bias.fill!(1)
  close(final_unmasked(sequence)[0], ones([2, 4, 4], device:) * 0.76159416, "last layer is never dropped")
  time_first = GRU(3, 4, num_layers: 2, bidirectional: true, batch_first: false, device:)
  time_first.load_state_dict!(clean.state_dict)
  close(time_first(sequence.transpose(0, 1).snapshot)[0], clean(sequence)[0].transpose(0, 1), "noncontiguous read-only time-first input")

  projected = LSTMCell(3, 4, proj_size: 2, bias: false, device:)
  (h, c) = projected(x[0])
  check(h.shape == [2] and c.shape == [4] and projected.parameters.count == 3, "projected unbatched LSTMCell")
  projected(x[0], state: (h, c))
  check(RNNCell(3, 4, bias: false, device:).parameters.count == 2, "RNNCell bias disabled")
  check(GRUCell(3, 4, bias: false, device:).parameters.count == 2, "GRUCell bias disabled")
  rejected(): RNN(3, 4, num_layers: 0, device:)
  rejected(): GRU(3, 4, num_layers: 1.5, device:)
  rejected(): RNN(3, 4, nonlinearity: :invalid, device:)
  rejected(): LSTM(3, 4, proj_size: 4, device:)
  rejected(): LSTM(3, 4, proj_size: -1, device:)
  rejected(): LSTMCell(3, 4, proj_size: 1.5, device:)
  rejected(): GRU(3, 4, dropout: -0.1, device:)
  rejected(): GRU(3, 4, device:)(zeros([2, 0, 3], device:))
  rejected(): GRU(3, 4, device:)(zeros([2, 4, 2], device:))
  rejected(): GRU(3, 4, device:)(sequence, state: zeros([1, 4], device:))
  rejected(): projected(x[0], state: (zeros([4], device:), zeros([4], device:)))
  rejected(): projected(x[0], state: (h, c.to(device:, dtype: "int32")))
  if device == "mps":
    rejected(): GRU(3, 4, device:)(sequence, state: zeros([1, 2, 4]))
  detached = no_grad(): clean(sequence)[0]
  inferred = inference_mode(): clean(sequence)[0]
  check(not detached.requires_grad? and not inferred.requires_grad?, "recurrent execution scopes")
  print "PASS torch recurrent " + device
  0

def main():
  parser = ArgParser(name: "Recurrent checks")
  parser.arg("--device", choices: ["cpu", "mps"], default: "cpu")
  run(parser.parse_or_raise["device"])
