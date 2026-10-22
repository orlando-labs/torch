package torch.nn.utils.sequence

from torch.tensor.value import Tensor
from torch.tensor.factories import tensor
from torch.nn.utils.packed_sequence import PackedSequence
from torch.autograd.context import execution_mode
from torch.errors import ShapeError, DTypeError, DeviceError

export pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence, unpad_sequence, unpack_sequence
export packed_padded_sequence, padded_packed_sequence, packed_sequence, padded_sequence, unpadded_sequence, unpacked_sequence

native def _sequence(op as Str, inputs, options, mode as Int) from "amt_sequence" blocking

def _kernel(op, inputs, options, pair: false):
  handles = inputs.map: $it._core
  result = _sequence(op, handles, options, execution_mode())
  if pair:
    result.map: Tensor($it)
  else:
    Tensor(result)

def _lengths(input, lengths, batch_first, allow_zero):
  raise TypeError("padded input must be a Tensor") unless Tensor === input
  raise ShapeError("padded input requires rank >= 2") if input.rank < 2
  if Tensor === lengths:
    raise DeviceError("lengths Tensor must be on CPU") unless lengths.device.type == "cpu"
    raise DTypeError("lengths Tensor must be integral") unless lengths.dtype.integral?
    raise ShapeError("lengths Tensor must be a vector") unless lengths.rank == 1
    lengths = lengths.to_a
  else:
    raise TypeError("lengths must be an Array, Tuple or CPU Tensor") unless Array === lengths or Tuple === lengths
    lengths = lengths.array
  batch = if batch_first then input.shape[0] else input.shape[1]
  time = if batch_first then input.shape[1] else input.shape[0]
  raise ShapeError("lengths must match the batch size") unless lengths.count == batch
  lengths.each |length|:
    unless Int === length and length <= time and (length > 0 or allow_zero and length == 0):
      raise ValueError("sequence lengths must fit the padded time dimension")
  lengths

def pack_padded_sequence(input, lengths, batch_first: false, enforce_sorted: true):
  lengths = _lengths(input, lengths, batch_first, false)
  raise ValueError("cannot pack an empty batch") if lengths.empty?
  sorted_indices = null
  if enforce_sorted:
    lengths.each_with_index |length, index|:
      raise ValueError("enforce_sorted requires nonincreasing lengths") if index > 0 and lengths[index - 1] < length
  else:
    indices = Array.of(lengths.count): $it
    indices = indices.sorted: -lengths[$it]
    lengths = indices.map: lengths[$it]
    sorted_indices = tensor(indices, dtype: "int64", device: input.device)
    input = input.index_select(if batch_first then 0 else 1, sorted_indices)
  values = _kernel("pack", [input, tensor(lengths, dtype: "int64")], [batch_first], pair: true)
  PackedSequence(values[0], values[1], sorted_indices:)

def pad_packed_sequence(sequence, batch_first: false, padding_value: 0.0, total_length: null):
  raise TypeError("pad_packed_sequence requires PackedSequence") unless PackedSequence === sequence
  longest = sequence.batch_sizes.numel
  total_length = longest if total_length == null
  raise ValueError("total_length must be an Int >= packed sequence length") unless Int === total_length and total_length >= longest
  values = _kernel("unpack", [sequence.data, sequence.batch_sizes], [batch_first, padding_value, total_length], pair: true)
  (output, lengths) = values.tuple
  if sequence.unsorted_indices != null:
    output = output.index_select(if batch_first then 0 else 1, sequence.unsorted_indices)
    lengths = lengths.index_select(0, sequence.unsorted_indices.to(device: "cpu"))
  return (output, lengths)

def pad_sequence(sequences, batch_first: false, padding_value: 0.0, padding_side: :right):
  raise TypeError("sequences must be an Array or Tuple") unless Array === sequences or Tuple === sequences
  sequences = sequences.array
  raise ValueError("sequences must be nonempty") if sequences.empty?
  raise ValueError("padding_side must be :left or :right") unless padding_side.str in ["left", "right"]
  sequences.each |x|:
    raise TypeError("sequences must contain Tensors") unless Tensor === x
    raise ShapeError("sequences must have a time axis") if x.rank < 1
    unless x.shape.drop(1) == sequences[0].shape.drop(1):
      raise ShapeError("sequence trailing shapes must match")
    raise DTypeError("sequence dtypes must match") unless x.dtype == sequences[0].dtype
    raise DeviceError("sequence devices must match") unless x.device.to_str == sequences[0].device.to_str
  _kernel("pad", sequences, [batch_first, padding_value, padding_side.str])

def pack_sequence(sequences, enforce_sorted: true):
  padded = pad_sequence(sequences)
  lengths = sequences.map: $it.shape[0]
  pack_padded_sequence(padded, lengths, enforce_sorted:)

def unpad_sequence(padded_sequences, lengths, batch_first: false, padding_side: :right):
  lengths = _lengths(padded_sequences, lengths, batch_first, true)
  raise ValueError("padding_side must be :left or :right") unless padding_side.str in ["left", "right"]
  input = if batch_first then padded_sequences else padded_sequences.transpose(0, 1)
  Array.of(lengths.count) |index|:
    length = lengths[index]
    start = if padding_side.str == "right" then 0 else input.shape[1] - length
    input[index].narrow(0, start, length)

def unpack_sequence(sequence):
  (padded, lengths) = pad_packed_sequence(sequence, batch_first: true)
  unpad_sequence(padded, lengths, batch_first: true)

def packed_padded_sequence(input, lengths, batch_first: false, enforce_sorted: true):
  pack_padded_sequence(input, lengths, batch_first:, enforce_sorted:)

def padded_packed_sequence(sequence, batch_first: false, padding_value: 0.0, total_length: null):
  pad_packed_sequence(sequence, batch_first:, padding_value:, total_length:)

def packed_sequence(sequences, enforce_sorted: true): pack_sequence(sequences, enforce_sorted:)

def padded_sequence(sequences, batch_first: false, padding_value: 0.0, padding_side: :right):
  pad_sequence(sequences, batch_first:, padding_value:, padding_side:)

def unpadded_sequence(padded_sequences, lengths, batch_first: false, padding_side: :right):
  unpad_sequence(padded_sequences, lengths, batch_first:, padding_side:)

def unpacked_sequence(sequence): unpack_sequence(sequence)
