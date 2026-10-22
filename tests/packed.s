package torch.packed_test

from torch import tensor, randn, zeros, ones
from torch.tensor.value import Tensor
from torch.nn import RNN, GRU, LSTM, LSTMCell, Sequential
import torch.nn.utils.rnn as sequences
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pack_sequence, pad_sequence, unpad_sequence, unpack_sequence
from torch.autograd import no_grad, inference_mode, grad
from torch.checks import check

export main, run

native def _reference(kind, input, parameters, state, bias, layers, bidirectional, batch_first, sizes) from "amt_recurrent_reference" blocking
native def _sequence_reference(op, inputs, options) from "amt_sequence_reference" blocking

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
  check(failed, "invalid packed call must raise")

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

def model_for(kind, device, dtype, bias, bidirectional, layers, projection = 0, dropout = 0):
  case! kind:
    when "lstm": LSTM(3, 4, num_layers: layers, bias:, bidirectional:, batch_first: false,
      proj_size: projection, dropout:, seed: 10, device:, dtype:)
    when "gru": GRU(3, 4, num_layers: layers, bias:, bidirectional:, dropout:, seed: 10, device:, dtype:)
    else: RNN(3, 4, num_layers: layers, bias:, bidirectional:, nonlinearity: kind,
      dropout:, seed: 10, device:, dtype:)

def compare_recurrent(kind, device, dtype, bias, bidirectional, layers, projection = 0):
  model = model_for(kind, device, dtype, bias, bidirectional, layers, projection)
  no_grad():
    model.cells.each |cell|:
      cell.input_projection.bias.fill!(0.04) if bias
      cell.hidden_projection.bias.fill!(-0.03) if bias
  x = randn([4, 6, 3], seed: 1, device:, dtype:, requires_grad: true)
  packed = pack_padded_sequence(x, [2, 5, 1, 3], batch_first: true, enforce_sorted: false)
  directions = if bidirectional then 2 else 1
  width = if projection == 0 then 4 else projection
  h = randn([layers * directions, 4, width], seed: 2, device:, dtype:, requires_grad: true)
  c = randn([layers * directions, 4, 4], seed: 3, device:, dtype:, requires_grad: true)
  sorted_h = h.index_select(1, packed.sorted_indices)
  sorted_c = c.index_select(1, packed.sorted_indices)
  states = if kind == "lstm" then [sorted_h._core, sorted_c._core] else [sorted_h._core]
  parameters = weights(model)
  handles = parameters.map: $it._core
  expected = _reference(kind, packed.data._core, handles, states, bias, layers, bidirectional, false, packed.batch_sizes._core)
  state = if kind == "lstm" then (h, c) else h
  (output, final) = model(packed, state:)
  hidden = if kind == "lstm" then final[0] else final
  inverse = packed.unsorted_indices.to(device: "cpu")
  close(output.data, Tensor(expected[0]), "ATen packed output " + kind)
  close(hidden, Tensor(expected[1]).index_select(1, inverse), "ATen packed final hidden " + kind)
  close(output.batch_sizes, packed.batch_sizes, "packed output layout")
  close(output.sorted_indices, packed.sorted_indices, "packed sorting metadata preserved")
  loss = output.data.square.sum + hidden.square.sum
  if kind == "lstm":
    close(final[1], Tensor(expected[2]).index_select(1, inverse), "ATen packed final memory")
    loss = loss + final[1].square.sum
  packed_gradient = grad([loss], [packed.data], retain_graph: true)[0]
  close(packed_gradient, Tensor(expected[3]), "ATen packed data gradient")
  loss.backward!
  close(h.grad, Tensor(expected[4]).index_select(1, inverse), "ATen packed initial hidden gradient")
  close(c.grad, Tensor(expected[5]).index_select(1, inverse), "ATen packed initial memory gradient") if kind == "lstm"
  parameters.each_with_index |parameter, index|:
    close(parameter.grad, Tensor(expected[6][index]), "ATen packed parameter gradient")
    check(parameter.grad.device.type == device, "packed parameter gradient backend")
  [2, 5, 1, 3].each_with_index |length, index|:
    close(x.grad[index].narrow(0, length, 6 - length), zeros([6 - length, 3], device:, dtype:), "padding has zero gradient")
  check(output.data.device.type == device and hidden.device.type == device, "packed output backend")

def utilities(device):
  x = tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]], device:)
  packed = sequences.packed_padded_sequence(x, [2, 1, 3], batch_first: true, enforce_sorted: false)
  check(packed.data.to_a == [4, 1, 3, 5, 2, 6], "known packed data order")
  check(packed.batch_sizes.to_a == [3, 2, 1], "known batch sizes")
  check(packed.sorted_indices.to_a == [2, 0, 1] and packed.unsorted_indices.to_a == [1, 2, 0], "known sorting permutations")
  (padded, lengths) = sequences.padded_packed_sequence(packed, batch_first: true, total_length: 5, padding_value: -1)
  check(padded.to_a == [[1, 2, -1, -1, -1], [3, -1, -1, -1, -1], [4, 5, 6, -1, -1]], "padding value and total length")
  check(lengths.to_a == [2, 1, 3] and lengths.device.type == "cpu", "unpadded lengths restore original order on CPU")
  check(packed.batch_sizes.device.type == "cpu" and packed.batch_sizes.dtype.to_str == "int64", "CPU metadata invariant")
  check(packed.sorted_indices.device.type == device and packed.unsorted_indices.device.type == device, "permutation device invariant")
  check(packed.batch_sizes.readonly? and packed.sorted_indices.readonly?, "packing owns protected metadata")
  check(packed.to(device:) == packed, "same-device PackedSequence transfer is identity")
  converted = packed.to(device:, dtype: "float32")
  check(converted.data.dtype.to_str == "float32" and converted.batch_sizes.dtype.to_str == "int64", "transfer casts data only")
  restored = converted.to(device: "cpu")
  check(restored.sorted_indices.device.type == "cpu", "transfer moves permutations with data")

  [false, true].each |batch_first|:
    input = randn([3, 4, 2], seed: 2, device:, requires_grad: true)
    shaped = if batch_first then input else input.transpose(0, 1)
    lens = tensor([4, 3, 1], dtype: "int64")
    expected = _sequence_reference("pack", [shaped._core, lens._core], [batch_first])
    packed = pack_padded_sequence(shaped, lens, batch_first:)
    close(packed.data, Tensor(expected[0]), "ATen packing layout")
    close(packed.batch_sizes, Tensor(expected[1]), "ATen packing batch sizes")
    packed.data.square.sum.backward!
    wanted = Tensor(expected[2][0])
    wanted = wanted.transpose(0, 1) unless batch_first
    close(input.grad, wanted, "ATen packing gradient")
    check(packed.sorted_indices == null and packed.unsorted_indices == null, "sorted input has no permutation")
    data = packed.data.detach.copy.requires_grad!(true)
    sequence = PackedSequence(data, packed.batch_sizes)
    expected = _sequence_reference("unpack", [data._core, sequence.batch_sizes._core], [batch_first, -0.5, 6])
    (output, lengths) = pad_packed_sequence(sequence, batch_first:, padding_value: -0.5, total_length: 6)
    close(output, Tensor(expected[0]), "ATen unpacking values")
    close(lengths, Tensor(expected[1]), "ATen unpacking lengths")
    output.square.sum.backward!
    close(data.grad, Tensor(expected[2][0]), "ATen unpacking gradient")
    [:left, :right].each |padding_side|:
      items = [randn([3, 2], seed: 5, device:, requires_grad: true),
        randn([1, 2], seed: 6, device:, requires_grad: true), zeros([0, 2], device:, requires_grad: true)]
      handles = items.map: $it._core
      expected = _sequence_reference("pad", handles, [batch_first, 0.75, padding_side.str])
      padded = sequences.padded_sequence(items, batch_first:, padding_value: 0.75, padding_side:)
      close(padded, Tensor(expected[0]), "ATen left/right pad_sequence")
      padded.square.sum.backward!
      items.each_with_index |item, index|: close(item.grad, Tensor(expected[2][index]), "ATen pad_sequence gradient")
      unpadded = sequences.unpadded_sequence(padded, [3, 1, 0], batch_first:, padding_side:)
      items.each_with_index |item, index|: close(unpadded[index], item, "unpad left/right empty sequences")
  items = [tensor([1, 2], device:), tensor([3], device:), tensor([4, 5, 6], device:)]
  packed = sequences.packed_sequence(items, enforce_sorted: false)
  unpacked = sequences.unpacked_sequence(packed)
  items.each_with_index |item, index|: close(unpacked[index], item, "pack/unpack sequence round trip")
  immutable = ones([2, 3, 2], device:).snapshot
  sizes = tensor([3, 1], dtype: "int64")
  pack_padded_sequence(immutable, sizes, batch_first: true)
  close(immutable, ones(immutable.shape, device:), "packing read-only input does not mutate")
  manual_sizes = tensor([2, 1], dtype: "int64")
  manual_order = tensor([1, 0], dtype: "int64", device:)
  manual = PackedSequence(ones([3, 2], device:), manual_sizes, sorted_indices: manual_order)
  manual_sizes[0] = 1
  manual_order[0] = 0
  check(manual.batch_sizes.to_a == [2, 1] and manual.sorted_indices.to_a == [1, 0], "external metadata mutation cannot change packed layout")
  tied = pack_padded_sequence(ones([3, 3, 2, 2], device:), (2, 3, 2), batch_first: true, enforce_sorted: false)
  check(tied.sorted_indices.to_a == [1, 0, 2] and tied.data.shape == [7, 2, 2], "stable ties and multiple trailing dimensions")
  (restored, lengths) = pad_packed_sequence(tied, batch_first: true)
  check(restored.shape == [3, 3, 2, 2] and lengths.to_a == [2, 3, 2], "trailing dimensions and Tuple lengths restore")
  pair = (ones([2, 2], device:), ones([1, 2], device:))
  check(unpack_sequence(pack_sequence(pair)).count == 2, "Tuple sequence input")
  check(unpad_sequence(zeros([0, 3, 2], device:), [], batch_first: true).empty?, "unpad empty batch")
  [false, true].each |batch_first|:
    [:left, :right].each |padding_side|:
      input = ones([2, 3, 2], device:, requires_grad: true)
      shaped = if batch_first then input else input.transpose(0, 1)
      parts = unpad_sequence(shaped, [3, 1], batch_first:, padding_side:)
      (parts[0].sum + parts[1].sum).backward!
      expected = if padding_side.str == "right":
        [[[1, 1], [1, 1], [1, 1]], [[1, 1], [0, 0], [0, 0]]]
      else:
        [[[1, 1], [1, 1], [1, 1]], [[0, 0], [0, 0], [1, 1]]]
      close(input.grad, tensor(expected, dtype: "float32", device:), "unpad gradient excludes padding")
      close(input, ones(input.shape, device:), "unpad never mutates source")

def run(device):
  utilities(device)
  ["tanh", "relu", "gru", "lstm"].each |kind|:
    compare_recurrent(kind, device, "float32", true, true, 2)
    compare_recurrent(kind, device, "float32", false, false, 1)
  compare_recurrent("lstm", device, "float32", true, true, 2, 2)
  compare_recurrent("gru", device, "float64", true, true, 2) if device == "cpu"
  packed = pack_sequence([randn([2, 3], seed: 1, device:), randn([5, 3], seed: 2, device:), randn([1, 3], seed: 3, device:)], enforce_sorted: false)
  ["tanh", "gru", "lstm"].each |kind|:
    model = model_for(kind, device, "float32", true, true, 2)
    (output, final) = model(packed)
    unbatched = unpack_sequence(output)
    source = unpack_sequence(packed)
    3.times |index|:
      (dense, dense_final) = model(source[index])
      close(unbatched[index], dense, "packed versus independent dense sequence")
      if kind == "lstm":
        close(final[0].select(1, index), dense_final[0], "packed hidden matches dense terminal state")
        close(final[1].select(1, index), dense_final[1], "packed memory matches dense terminal state")
      else:
        close(final.select(1, index), dense_final, "packed state matches dense terminal state")
    restored = model_for(kind, device, "float32", true, true, 2)
    restored.load_state_dict!(model.state_dict)
    close(restored(packed)[0].data, output.data, "packed state dict round trip")
    continued = model(packed, state: final)
    check(continued[0].data.shape == output.data.shape, "packed explicit state continuation")
  noisy = GRU(3, 4, num_layers: 2, bidirectional: true, dropout: 0.5, device:)
  before = noisy.rng_state
  rejected(): noisy(packed)
  close(noisy.rng_state, before, "packed pure call does not consume RNG")
  first = noisy.call!(packed)[0].data
  noisy.load_rng_state!(before)
  close(noisy.call!(packed)[0].data, first, "packed dropout RNG replay")
  before_invalid = noisy.rng_state
  rejected(): noisy.call!(packed, state: zeros([4, 2, 4], device:))
  close(noisy.rng_state, before_invalid, "packed invalid state does not consume RNG")
  rejected(): Sequential([noisy])(packed)
  noisy.eval!
  clean = GRU(3, 4, num_layers: 2, bidirectional: true, device:)
  close(noisy(packed)[0].data, clean(packed)[0].data, "packed eval disables dropout")
  close(noisy.rng_state, before_invalid, "packed eval preserves RNG")
  dropped = RNN(3, 4, num_layers: 2, dropout: 1, bias: false, device:)
  close(dropped(packed)[0].data, zeros([8, 4], device:), "packed dropout p1")
  detached = no_grad(): clean(packed)[0].data
  inferred = inference_mode(): clean(packed)[0].data
  check(not detached.requires_grad? and not inferred.requires_grad?, "packed grad scopes")
  x = zeros([2, 3, 3], device:)
  rejected(): pack_padded_sequence(x, [1, 3], batch_first: true)
  rejected(): pack_padded_sequence(x, [0, 3], batch_first: true, enforce_sorted: false)
  rejected(): pack_padded_sequence(x, [4, 3], batch_first: true)
  rejected(): pack_padded_sequence(x, [3], batch_first: true)
  rejected(): pack_padded_sequence(x, [3.0, 1], batch_first: true)
  rejected(): pad_packed_sequence(packed, total_length: 1)
  rejected(): PackedSequence(ones([3, 3], device:), tensor([1, 2], dtype: "int64"))
  rejected(): PackedSequence(ones([3, 3], device:), tensor([2, 0, 1], dtype: "int64"))
  rejected(): PackedSequence(ones([4, 3], device:), tensor([2, 1], dtype: "int64"))
  rejected(): PackedSequence(ones([3, 3], device:), tensor([2, 1], dtype: "int32"))
  rejected(): PackedSequence(ones([3, 3], device:), tensor([2, 1], dtype: "int64"), sorted_indices: tensor([0, 0], dtype: "int64", device:))
  rejected(): PackedSequence(ones([3, 3], device:), tensor([2, 1], dtype: "int64"), sorted_indices: tensor([1, 0], dtype: "int64", device:), unsorted_indices: tensor([0, 1], dtype: "int64", device:))
  rejected(): pack_sequence([])
  rejected(): pad_sequence([ones([2, 3], device:), ones([2, 4], device:)])
  rejected(): unpad_sequence(x, [3, 1], batch_first: true, padding_side: :bad)
  rejected(): GRU(2, 4, device:)(packed)
  if device == "mps":
    rejected(): pack_padded_sequence(x, tensor([3, 1], dtype: "int64", device:), batch_first: true)
    rejected(): PackedSequence(ones([3, 3], device:), tensor([2, 1], dtype: "int64", device:))
  print "PASS torch packed sequences " + device
  0

def main():
  parser = ArgParser(name: "Packed sequence checks")
  parser.arg("--device", choices: ["cpu", "mps"], default: "cpu")
  run(parser.parse_or_raise["device"])
