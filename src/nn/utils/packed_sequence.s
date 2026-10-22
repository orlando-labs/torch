package torch.nn.utils.packed_sequence

from torch.tensor.value import Tensor
from torch.tensor.factories import tensor
from torch.errors import ShapeError, DTypeError, DeviceError

export PackedSequence

class PackedSequence:
  attr data
  attr batch_sizes
  attr sorted_indices
  attr unsorted_indices

  def init(@data, batch_sizes, sorted_indices: null, unsorted_indices: null):
    raise TypeError("packed data must be a Tensor") unless Tensor === data
    raise ShapeError("packed data requires rank >= 1") if data.rank < 1
    raise TypeError("batch_sizes must be a Tensor") unless Tensor === batch_sizes
    unless batch_sizes.dtype.to_str == "int64" and batch_sizes.device.type == "cpu":
      raise DTypeError("batch_sizes must be a CPU int64 Tensor")
    raise ShapeError("batch_sizes must be a nonempty vector") unless batch_sizes.rank == 1 and batch_sizes.numel > 0
    sizes = batch_sizes.to_a
    sizes.each_with_index |size, index|:
      raise ShapeError("batch_sizes must be positive and nonincreasing") unless size > 0 and (index == 0 or sizes[index - 1] >= size)
    raise ShapeError("packed row count must equal sum(batch_sizes)") unless data.shape[0] == sizes.sum
    @batch_sizes = batch_sizes.snapshot
    @sorted_indices = null
    @unsorted_indices = null
    if sorted_indices == null:
      raise ValueError("unsorted_indices requires sorted_indices") unless unsorted_indices == null
    else:
      order = _permutation(sorted_indices, sizes[0])
      inverse = Array.filled(sizes[0], 0)
      order.each_with_index |value, index|:
        inverse[value] = index
      if unsorted_indices != null:
        unless _permutation(unsorted_indices, sizes[0]) == inverse:
          raise ValueError("unsorted_indices must invert sorted_indices")
      else:
        unsorted_indices = tensor(inverse, dtype: "int64", device: data.device)
      @sorted_indices = sorted_indices.snapshot
      @unsorted_indices = unsorted_indices.snapshot

  def _permutation(indices, batch):
    raise TypeError("packed indices must be Tensors") unless Tensor === indices
    raise DTypeError("packed indices must be int64") unless indices.dtype.to_str == "int64"
    raise DeviceError("packed indices must share the data device") unless indices.device.to_str == @data.device.to_str
    raise ShapeError("packed indices must be batch-sized vectors") unless indices.shape == [batch]
    # This is explicit metadata validation, never a copy of sequence data.
    values = indices.to(device: "cpu").to_a
    in_bounds = values.all?: $it >= 0 and $it < batch
    unless values.uniq.count == batch and in_bounds:
      raise ValueError("packed indices must be permutations of the batch")
    values

  def to(device:, dtype: "same"):
    converted = @data.to(device:, dtype:)
    return self if converted._identity == @data._identity
    sorted = if @sorted_indices == null then null else @sorted_indices.to(device: converted.device)
    unsorted = if @unsorted_indices == null then null else @unsorted_indices.to(device: converted.device)
    PackedSequence(converted, @batch_sizes, sorted_indices: sorted, unsorted_indices: unsorted)
