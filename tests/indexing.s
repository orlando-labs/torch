package torch.indexing_test

from torch import tensor, zeros, ones, Slice, no_grad
import torch.tensor.indexing as ops
from torch.checks import check

export main, run

def same(x, values, message): check(x.to_a == values, message)

def rejected(&block):
  failed = false
  try:
    block()
  rescue Torch.Errors.Error:
    failed = true
  check(failed, "invalid indexing must raise")

def run(device):
  x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device:)
  i = tensor([[2, 0], [1, 1]], dtype: "int64", device:)
  rows = tensor([1, 0], dtype: "int64", device:)
  cols = tensor([2, 1], dtype: "int64", device:)
  mask = x.gt(3)
  same(x[-1], [4.0, 5.0, 6.0], "negative basic index")
  same(x.index([rows, cols]), [6.0, 2.0], "paired advanced indices")
  same(x[(rows, cols)], [6.0, 2.0], "tuple bracket indices")
  same(x[[1, 0]], [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], "Array advanced index")
  same(x[[true, false]], [[1.0, 2.0, 3.0]], "Bool Array index")
  same(x[mask], [4.0, 5.0, 6.0], "Bool Tensor index")
  same(x[[[false, false, false], [true, true, true]]], [4.0, 5.0, 6.0], "nested Bool Array index")
  same(x.index([Slice(stop: 1), Slice(start: 1)]), [[2.0, 3.0]], "basic slices")
  same(x.index([:ellipsis, -1]), [3.0, 6.0], "ellipsis")
  check(x.index([null, :all]).shape == [1, 2, 3], "new axis")
  same(x.indexed([rows, cols]), [6.0, 2.0], "indexed synonym")
  same(x.index_selected(0, rows), [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], "index_selected synonym")
  same(x.selected(0, 1), [4.0, 5.0, 6.0], "selected view synonym")
  check(x[true].shape == [1, 2, 3] and x[false].shape == [0, 2, 3], "Bool scalar index")
  check(x[[]].shape == [0, 3], "empty advanced index")
  check(x.index([Slice(start: 0, stop: 0)]).shape == [0, 3], "empty slice")
  same(x.index([Slice(), Slice(step: 2)]), [[1.0, 3.0], [4.0, 6.0]], "strided slice")

  same(ops.gather(x, 1, i), [[3.0, 1.0], [5.0, 5.0]], "gather")
  same(x.gathered(-1, i), [[3.0, 1.0], [5.0, 5.0]], "gathered synonym")
  same(x.take(tensor([0, -1, 3], dtype: "int64", device:)), [1.0, 6.0, 4.0], "flattened take")
  same(x.take(tensor([[0, 5], [2, 3]], dtype: "int64", device:)),
       [[1.0, 6.0], [3.0, 4.0]], "take preserves index shape")
  take_input = x.snapshot
  scalar_take = take_input.take(tensor(0, dtype: "int64", device:))
  check(scalar_take.shape == [] and not scalar_take.readonly?, "scalar take owns storage")
  check(scalar_take.item == 1.0, "scalar take value")
  scalar_take.fill!(9)
  same(take_input, x.to_a, "scalar take cannot mutate its input")
  same(x.take_along_dim(i, dim: 1), [[3.0, 1.0], [5.0, 5.0]], "take along axis")
  same(x.take_along_dim(rows), [2.0, 1.0], "take along flattened input")
  same(x.masked_selected(mask), [4.0, 5.0, 6.0], "masked select")
  same(x.masked_filled(mask, 9), [[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]], "masked fill scalar")
  same(x.masked_fill(mask, tensor(8.0, device:)), [[1.0, 2.0, 3.0], [8.0, 8.0, 8.0]], "masked fill Tensor")
  values = tensor([7.0, 8.0, 9.0], device:)
  same(x.masked_scattered(mask, values), [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], "masked scatter")
  same(mask.nonzero, [[1, 0], [1, 1], [1, 2]], "nonzero matrix")
  nz = mask.nonzero(as_tuple: true)
  same(nz[0], [1, 1, 1], "nonzero row tuple")
  same(nz[1], [0, 1, 2], "nonzero column tuple")
  same(ops.argwhere(mask), [[1, 0], [1, 1], [1, 2]], "argwhere")

  base = zeros([2, 3], device:)
  src = tensor([[10.0, 20.0], [30.0, 40.0]], device:)
  unique = tensor([[2, 0], [1, 2]], dtype: "int64", device:)
  same(base.scattered(1, unique, src), [[20.0, 0.0, 10.0], [0.0, 30.0, 40.0]], "scatter source")
  same(base.scatter(1, unique, 5), [[5.0, 0.0, 5.0], [0.0, 5.0, 5.0]], "scatter scalar")
  same(base.scatter_added(1, i, src), [[20.0, 0.0, 10.0], [0.0, 70.0, 0.0]], "scatter add duplicates")
  same(base.scatter(1, i, src, reduce: :add), [[20.0, 0.0, 10.0], [0.0, 70.0, 0.0]], "legacy scatter Tensor reduction")
  same(base.scatter(1, i, 2, reduce: :add), [[2.0, 0.0, 2.0], [0.0, 4.0, 0.0]], "legacy scatter scalar reduction")
  same(base.scatter_reduced(1, i, src, :sum), [[20.0, 0.0, 10.0], [0.0, 70.0, 0.0]], "scatter sum")
  same(base.scatter_reduce(1, i, src, :mean, include_self: false),
       [[20.0, 0.0, 10.0], [0.0, 35.0, 0.0]], "scatter mean excludes self")
  [:prod, :amax, :amin].each |reduction|:
    reduced = base.scatter_reduce(1, i, src, reduction, include_self: false)
    expected = 30.0
    expected = 40.0 if reduction == :amax
    expected = 1200.0 if reduction == :prod
    check(reduced[1][1].item == expected, "scatter reduction family")
  same(base.index_added(0, rows, x, alpha: 2), [[8.0, 10.0, 12.0], [2.0, 4.0, 6.0]], "index add alpha")
  same(base.index_copied(0, rows, x), [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], "index copy")
  same(base.index_filled(0, rows, 7), [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]], "index fill")
  same(base.index_fill(0, rows, tensor(7.0, device:)), [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]], "index fill Tensor")
  same(base.index_reduced(0, rows, x, :mean, include_self: false),
       [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], "index reduce")
  same(base.index_put([rows, cols], 8), [[0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], "functional indexed scalar write")
  duplicates = tensor([0, 0], dtype: "int64", device:)
  same(zeros([3], device:).index_put([duplicates], tensor([2.0, 3.0], device:), accumulate: true),
       [5.0, 0.0, 0.0], "index put accumulates duplicates")
  same(zeros([3], device:).index_put([duplicates], 2, accumulate: true),
       [4.0, 0.0, 0.0], "index put accumulates scalar")
  same(base.put(duplicates, tensor([2.0, 3.0], device:), accumulate: true),
       [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "put accumulates duplicates")
  same(base.select_scatter(values, 0, 1), [[0.0, 0.0, 0.0], [7.0, 8.0, 9.0]], "select scatter")
  same(base.slice_scatter(x, dim: 1), x.to_a, "slice scatter")
  same(base.diagonal_scatter(tensor([7.0, 8.0], device:)),
       [[7.0, 0.0, 0.0], [0.0, 8.0, 0.0]], "diagonal scatter")
  same(base.put(tensor([0, 5], dtype: "int64", device:), tensor([7.0, 8.0], device:)),
       [[7.0, 0.0, 0.0], [0.0, 0.0, 8.0]], "put")
  same(base, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "all non-bang writes preserve receiver")

  target = base.copy
  check(target.scatter!(1, unique, src) == target, "bang returns self")
  same(target, base.scatter(1, unique, src).to_a, "scatter bang")
  target.zero!.scatter_add!(1, i, src)
  same(target, base.scatter_add(1, i, src).to_a, "scatter add bang")
  target.zero!.scatter_reduce!(1, i, src, :mean, include_self: false)
  same(target, base.scatter_reduce(1, i, src, :mean, include_self: false).to_a, "scatter reduce bang")
  target.zero!.index_add!(0, rows, x, alpha: 2)
  same(target, base.index_add(0, rows, x, alpha: 2).to_a, "index add bang")
  target.zero!.index_copy!(0, rows, x)
  same(target, base.index_copy(0, rows, x).to_a, "index copy bang")
  target.zero!.index_fill!(0, rows, 7)
  same(target, base.index_fill(0, rows, 7).to_a, "index fill bang")
  target.zero!.index_reduce!(0, rows, x, :mean, include_self: false)
  same(target, base.index_reduce(0, rows, x, :mean, include_self: false).to_a, "index reduce bang")
  target.zero!.masked_fill!(mask, 9)
  same(target, base.masked_fill(mask, 9).to_a, "masked fill bang")
  target.zero!.masked_scatter!(mask, values)
  same(target, base.masked_scatter(mask, values).to_a, "masked scatter bang")
  target.zero!.put!(tensor([0, 5], dtype: "int64", device:), tensor([7.0, 8.0], device:))
  same(target, base.put(tensor([0, 5], dtype: "int64", device:), tensor([7.0, 8.0], device:)).to_a, "put bang")
  transposed_base = zeros([2, 3], device:)
  transposed_base.transpose(0, 1).put!(tensor([0, 5], dtype: "int64", device:), tensor([7.0, 8.0], device:))
  same(transposed_base, [[7.0, 0.0, 0.0], [0.0, 0.0, 8.0]], "noncontiguous put bang")
  target.zero!
  target[(1, Slice())] = values
  same(target, [[0.0, 0.0, 0.0], [7.0, 8.0, 9.0]], "bracket assignment")
  target.index([Slice(), 1]).fill!(2)
  same(target, [[0.0, 2.0, 0.0], [7.0, 2.0, 9.0]], "basic index is a writable view")
  target[rows].zero!
  same(target, [[0.0, 2.0, 0.0], [7.0, 2.0, 9.0]], "advanced index is an independent copy")
  view = target.transpose(0, 1)
  view.index_put!([1, Slice()], 4)
  same(target, [[0.0, 4.0, 0.0], [7.0, 4.0, 9.0]], "noncontiguous indexed write")

  frozen = x.snapshot
  check(frozen.index([Slice()]).readonly?, "frozen basic view remains read-only")
  check(not frozen[rows].readonly?, "advanced selection owns writable storage")
  frozen[rows].zero!
  same(frozen, x.to_a, "advanced copy cannot mutate snapshot")
  denied = false
  try:
    frozen.index_put!([0], 9)
  rescue Torch.Errors.ReadOnlyTensorError:
    denied = true
  check(denied, "read-only indexed write")
  denied = false
  try:
    frozen.masked_fill!(mask, 9)
  rescue Torch.Errors.ReadOnlyTensorError:
    denied = true
  check(denied, "read-only masked write")
  rejected(): target.index_copy!(0, rows, target)
  rejected(): target.gather(1, rows)
  rejected(): target.gather(1, tensor([[3], [0]], dtype: "int64", device:))
  rejected(): target.gather(1, tensor([[0.0], [1.0]], device:))
  rejected(): target[[1.0]]
  rejected(): target.take(tensor([0.0], device:))
  rejected(): target.index_put!([tensor([3], dtype: "int64", device:)], 9)
  rejected(): target.scatter!(1, tensor([[3], [0]], dtype: "int64", device:), tensor([[1.0], [1.0]], device:))
  same(target, [[0.0, 4.0, 0.0], [7.0, 4.0, 9.0]], "invalid writes preserve values")

  leaf = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device:, requires_grad: true)
  rejected(): leaf.index_put!([0], 1)
  leaf.gathered(1, i).sum.backward!
  same(leaf.grad, [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]], "gather gradient duplicates")
  leaf.zero_grad!
  leaf[(rows, cols)].sum.backward!
  same(leaf.grad, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "advanced index autograd")
  leaf.zero_grad!
  leaf.masked_fill(mask, 0).sum.backward!
  same(leaf.grad, [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], "masked fill autograd")
  input = ones([2, 3], device:, requires_grad: true)
  source = tensor([[10.0, 20.0], [30.0, 40.0]], device:, requires_grad: true)
  input.scatter_add(1, i, source).sum.backward!
  same(input.grad, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], "scatter base gradient")
  same(source.grad, [[1.0, 1.0], [1.0, 1.0]], "scatter source gradient")
  leaf.zero_grad!
  patch = tensor([7.0, 8.0, 9.0], device:, requires_grad: true)
  leaf.index_put([0], patch).sum.backward!
  same(leaf.grad, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], "index put base gradient")
  same(patch.grad, [1.0, 1.0, 1.0], "index put source gradient")
  put_leaf = ones([3], device:, requires_grad: true)
  put_source = tensor([7.0, 8.0], device:, requires_grad: true)
  nonleaf = put_leaf * 2
  nonleaf.put!(tensor([0, 2], dtype: "int64", device:), put_source).sum.backward!
  same(put_leaf.grad, [0.0, 2.0, 0.0], "put bang base gradient")
  same(put_source.grad, [1.0, 1.0], "put bang source gradient")
  no_grad(): leaf.index_put!([0], 9)
  same(leaf[0], [9.0, 9.0, 9.0], "no-grad indexed leaf write")
  check(leaf.device.type == device and source.grad.device.type == device, "actual backend")
  check(x.index([rows, cols]).device.type == device and mask.nonzero.device.type == device,
        "index and nonzero backend")
  check(base.scatter_reduce(1, i, src, :sum).device.type == device, "scatter backend")
  print "PASS torch indexing " + device
  0

def main(): run("cpu")
