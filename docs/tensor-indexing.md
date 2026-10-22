# Tensor Indexing

The adapter delegates indexing to LibTorch 2.14.0 ATen kernels. Sputnik
normalizes public values and preserves explicit mutation and read-only
contracts. Python is not a runtime or test dependency.

## Access

```sputnik
from torch import tensor, Slice

x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
x[0]                              # first row, a view
x[[1, 0]]                         # reordered rows, an independent copy
x[x.gt(3)]                        # Bool mask, an independent copy
x.index([Slice(), 1])             # second column, a view
x[(Slice(stop: 1), Slice())]      # tuple specifies multiple axes
x.index([null, :ellipsis])        # new axis and remaining dimensions
x.index([:all, Slice(step: 2)])   # full first axis, strided second axis
```

`index` accepts an Array or Tuple of per-axis indices. `[]` accepts one
index; a Tuple inside the brackets specifies multiple axes, whereas an Array
is an integer/Bool advanced index on the first axis. Tensor indices retain
their dtype and placement; Array indices become owned CPU integer/Bool
tensors, which LibTorch indexing may use for a CPU or MPS receiver.

Supported indices are Int (including negatives), Bool, integer/Bool Tensor,
integer/Bool Array, `Slice`, `null` (new axis), `:all` and `:ellipsis`.
`Slice(start: null, stop: null, step: 1)` has an exclusive stop and a positive
step, matching LibTorch. Use Slice rather than a Sputnik Range; comma-separated
bracket indices are not part of the Sputnik grammar.

Basic indexing returns a storage alias and propagates read-only status.
Advanced indexing returns an independent, writable result, even for a
snapshot receiver. Both remain differentiable according to ATen.

`take` returns independent storage with the index Tensor's shape, including
a scalar result for a scalar index. It does not return a basic-index view.

In the verified LibTorch 2.14.0 build, MPS has no `take`/`put_` kernels.
Sputnik implements these operations on MPS by flattening, using advanced
indexing/indexed writes, and restoring the shape. Bang put copies the result
into the original storage after native preflight, retaining its strides and
identity. Tensor computations stay on MPS, with automatic CPU fallback
disabled. Bounds preflight reduces index tensors on their device and reads
the resulting scalar bounds before dispatch, so invalid indices raise a
rescuable error without poisoning the Metal command stream. These checks
introduce synchronization. Non-bang put may return contiguous storage for a
noncontiguous input.

## Operations

These methods also have function forms exported from `torch` and
`torch.tensor.indexing`:

| Torch Name | Options | Bang Variant |
| --- | --- | --- |
| index_put | accumulate: false | index_put! |
| gather | sparse_grad: false | none |
| take | flattened integer indices | none |
| take_along_dim | dim: null | none |
| put | accumulate: false | put! |
| scatter | Tensor/scalar source; reduce: null | scatter! |
| scatter_add | Tensor source, duplicate accumulation | scatter_add! |
| scatter_reduce | sum/prod/mean/amax/amin; include_self: true | scatter_reduce! |
| index_add | alpha: 1 | index_add! |
| index_copy | Tensor source | index_copy! |
| index_fill | scalar or scalar Tensor | index_fill! |
| index_reduce | prod/mean/amax/amin; include_self: true | index_reduce! |
| masked_select | Bool mask | none |
| masked_fill | scalar or scalar Tensor | masked_fill! |
| masked_scatter | Tensor source | masked_scatter! |
| nonzero | as_tuple: false | none |
| select_scatter | dim, index | none |
| slice_scatter | dim: 0, start: null, end: null, step: 1 | none |
| diagonal_scatter | offset: 0, dim1: 0, dim2: 1 | none |

`index_select` remains available; `argwhere` is the function form of
`nonzero(as_tuple: false)`. Tuple nonzero returns a Tuple of Tensor values.
ATen determines dtype/device support and raises a typed package error for
unsupported combinations. Sparse gradient materialization is not part of
the current dense Tensor API.

## Mutation And Names

Torch non-bang names keep their normal, non-mutating semantics. Bang methods
modify the receiver and return `self`; bracket assignment is an indexed bang
write. They reject read-only storage and detected internal/source overlap,
retain LibTorch's leaf/autograd restrictions, and are forbidden during ONNX
export. Non-bang indexed writes return independent storage.

```sputnik
y = x.scatter(dim, indices, source)
z = x.scattered(dim, indices, source)   # same operation as scatter
x.scatter!(dim, indices, source)       # receiver mutation
x[(0, Slice())] = row                  # explicit indexed mutation
```

Additional method names are synonyms, not copies with changed alias rules:

| Torch Method | Sputnik Synonym |
| --- | --- |
| index | indexed |
| gather | gathered |
| scatter | scattered |
| scatter_add | scatter_added |
| scatter_reduce | scatter_reduced |
| index_add | index_added |
| index_copy | index_copied |
| index_fill | index_filled |
| index_reduce | index_reduced |
| masked_select | masked_selected |
| masked_fill | masked_filled |
| masked_scatter | masked_scattered |
| index_select | index_selected |
| select | selected |

In particular, `selected` still returns a view. No artificial `putted` or
`taked` names are added.

Accumulating `index_put` accepts only Tensor/Array advanced indices.
With duplicate indices and `accumulate: false`, overwrite behavior has the
same limitations as LibTorch. Prefer accumulation or scatter reductions
when duplicates are meaningful. Legacy `scatter(reduce: :add/:multiply)`
is forwarded for compatibility; prefer `scatter_add` or `scatter_reduce`.
