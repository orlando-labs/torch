# Embedding Operators

The embedding block uses LibTorch 2.14.0 ATen kernels without Python.
Configuration, fixed-size bag preparation, initialization and max-norm
composition stay in Sputnik. Embedding and EmbeddingBag have separate public
class files. Weights are registered parameters and participate in state_dict.

## API

Import classes from `torch.nn` and functions from
`torch.nn.functional`.

`Embedding(num_embeddings, embedding_dim, ...)` and
`EmbeddingBag(num_embeddings, embedding_dim, ...)` accept:

- `padding_idx: null`, `max_norm: null`, `norm_type: 2.0`.
- `scale_grad_by_freq: false`, `sparse: false`.
- `weight: null`, `freeze: false`, `seed: 0`, `device: "cpu"`, `dtype: "float32"`.

EmbeddingBag additionally accepts `mode: :mean` and
`include_last_offset: false`. Modes are `sum`, `mean` and `max`, as Symbols
or strings. Max bags reject sparse gradients and frequency scaling.

`embedding(input, weight, ...)` preserves the shape of integer indices and
appends the embedding dimension. Indices are int32/int64 on the weight device.
`embedding_bag(input, weight, offsets = null, ...)` returns one row per bag:

- A 1D input requires 1D offsets, starting at zero and nondecreasing.
- With `include_last_offset: true`, the final offset is the input length
  and is not the start of another bag.
- A 2D input represents fixed-size bags, does not accept offsets, and ignores
  include_last_offset, following the upstream functional API.
- `per_sample_weights: null` accepts a floating Tensor matching the original
  index shape, weight dtype and device. Weights are supported only for sum bags.
- Empty bags produce zeros, including max bags with negative weights.
- Padding entries are excluded from aggregation, mean denominators and
  parameter gradients.

Null padding means no padding. Negative padding indices select from the end
of the vocabulary, so `padding_idx: -1` now means the last row, not the old
SputnikTorch sentinel for no padding. Random initialization and
`reset_parameters!` zero the selected padding row. Supplied weights preserve
their padding values.

Both classes provide `from_pretrained(weights, freeze: true, ...)`. They take
an owned detached copy and return a leaf parameter, including for read-only
source weights. Unlike an implicitly shared pretrained tensor, subsequent
parameter mutation cannot change the supplied source. Reset preserves the
parameter object and its requires-grad setting.

See the upstream [Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)
and [EmbeddingBag](https://docs.pytorch.org/docs/2.14/generated/torch.nn.EmbeddingBag.html)
contracts for aggregation and gradient options.

## Explicit Mutation

Max-norm changes only referenced weight rows, under no-grad. It uses Sputnik
composition of vector norms, row selection and indexed writes on the weight
device. Repeated indices receive the same normalized value; unreferenced rows
are unchanged. Renormalization does not contribute to the backward graph.

Non-bang `embedding` and `embedding_bag` reject a non-null max_norm. Use
`embedding!` or `embedding_bag!` for that configuration. Modules configured
with max_norm require `call!` or `forward!`, even in evaluation mode.
Invalid indices, offsets and incompatible gradient options are checked
before renormalization changes weight.

`embedding_renorm!(weight, input, max_norm, norm_type: 2.0)` mutates weight.
`embedding_renormed(...)` copies it first and leaves the source unchanged;
read-only snapshots are accepted by the pure form. Torch non-bang lookup
names remain the default pure forms.

## Sparse Gradients And Backends

Sparse COO gradients are supported on CPU for Embedding and sum/mean
EmbeddingBag. Sparse gradients cannot be combined with frequency scaling.
`Tensor.is_sparse` and `sparse?` inspect the layout; `to_dense` and `densified`
convert a sparse gradient to a fresh writable dense Tensor. On an already
dense read-only Tensor, conversion preserves read-only status.

Existing optimizers require dense gradients. Conversion is explicit:

```sputnik
layer.weight.grad!(layer.weight.grad.densified)
optimizer.step!
```

Sparse MPS configurations raise explicitly before lookup. Dense kernels and
max-norm composition run on actual MPS with automatic CPU fallback disabled.
Frequency-scaled bag gradients on MPS use an explicit Sputnik composition:
device-side index counts and a detached value plus a scaled weight derivative.
This preserves lookup values and per-sample derivatives, and avoids the
incorrect native MPS bag frequency-scaling result observed in the verified
LibTorch build. CPU uses ATen's bag gradient directly.
Index bounds and bag offset validation synchronize MPS and read scalar/index
metadata on CPU; aggregation and differentiation remain on the input backend.
This is not a CPU computation fallback.

## Verification

`tests/embeddings.s` covers all three modes, empty/CSR/fixed-size bags,
noncontiguous and read-only inputs, int32/int64, padding, per-sample derivatives,
frequency scaling, CPU sparse gradients and explicit optimizer conversion.
It also checks initialization/reset, owned pretrained leaf parameters,
state restore, selected-row max-norm, invalid-input non-mutation and execution
scopes. Float64 is checked on CPU only.

CPU and actual MPS are exercised in VM and full-native modes. The shared
full-native executable accepts `--device cpu` or `--device mps`; the MPS VM
entry imports the same checks. Native builds require full body coverage,
and executables are checked with an empty PATH for absence of Python runtime
libraries. MPS runs set `PYTORCH_ENABLE_MPS_FALLBACK=0` and verify placement
of results, parameters and gradients.

The verified shared native build compiles all 2059 graph code bodies, with
full body coverage and zero VM/native fallback bodies. Both device runs pass.

The compiler regression found during bag preparation covers tuple assignment
to scattered parameter registers, captured locals, original RHS preservation
and rescuable MatchError in VM and full-native modes. List/map rest assignment
is additionally checked in VM; the native compiler's preexisting rest-pattern
fallback is unchanged and is not used by this embedding block.
