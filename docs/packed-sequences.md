# Packed Sequences

`torch.nn.utils.rnn` exports `PackedSequence` and the six sequence
utilities below. Packing/padding use LibTorch 2.14.0 tensor primitives;
sorting, validation, permutation restoration and recurrent traversal stay in
Sputnik. No Python dependency is introduced.

## Utilities

- `pack_padded_sequence(input, lengths, batch_first: false, enforce_sorted: true)`.
- `pad_packed_sequence(sequence, batch_first: false, padding_value: 0.0, total_length: null)`.
- `pack_sequence(sequences, enforce_sorted: true)`.
- `pad_sequence(sequences, batch_first: false, padding_value: 0.0, padding_side: :right)`.
- `unpad_sequence(padded_sequences, lengths, batch_first: false, padding_side: :right)`.
- `unpack_sequence(sequence)`.

These new utilities use the upstream time-first default, independently of the
existing batch-first default on Sputnik recurrent modules. Padded inputs are
`[T, N, ...]` or `[N, T, ...]`; individual sequences are `[length, ...]`.
Trailing dimensions must agree when padding a list. Labels and integer data
can be packed, not only floating-point feature vectors.

Lengths accept an Array, Tuple or one-dimensional integral CPU Tensor. Packing
requires a nonempty batch and positive lengths no larger than the padded time
dimension. Data past a declared length is discarded and receives zero gradient.
With `enforce_sorted: true`, lengths must already be nonincreasing. With false,
Sputnik stably sorts the batch and records both permutations without mutating
the input or lengths.

Padding returns `(padded, lengths)`, restoring original batch order; lengths
remain CPU int64. `total_length` extends the padded time dimension and must
not be shorter than the longest packed sequence. `pad_sequence` accepts empty
individual sequences and left/right padding. `unpad_sequence` returns views,
does not mutate its source, and accepts zero lengths. Its left-padding option
is a Sputnik extension to the upstream right-padding unpad API.

All functions are non-mutating under their Torch default names. The grammatical
Sputnik synonyms are `packed_padded_sequence`, `padded_packed_sequence`,
`packed_sequence`, `padded_sequence`, `unpadded_sequence` and `unpacked_sequence`.

The shape/layout contracts follow the upstream
[pack_padded_sequence](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.rnn.pack_padded_sequence.html),
[pad_packed_sequence](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html),
[pack_sequence](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pack_sequence.html),
[pad_sequence](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html)
and [unpad_sequence](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.rnn.unpad_sequence.html)
APIs.

## PackedSequence

`PackedSequence(data, batch_sizes, sorted_indices: null, unsorted_indices: null)`
exposes these four read-only attributes. Data must have a leading packed-row
axis. Batch sizes must be a nonempty, positive, nonincreasing CPU int64 vector
whose sum equals the data row count. Permutations must be batch-sized int64
vectors on the data device. If sorted indices are supplied, the constructor
computes the inverse or verifies the supplied inverse.

Metadata is stored as owned read-only snapshots so external mutation cannot
invalidate a packed layout. Data retains its original differentiable Tensor
handle. This strict manual-construction validation is a Sputnik contract;
upstream recommends creating packed values through the utility functions.
Metadata validation can explicitly copy only permutation indices to CPU,
never sequence data.

`sequence.to(device:, dtype: "same")` returns self for an unchanged data
Tensor, otherwise transfers/casts data and moves permutations with it.
Batch sizes always remain CPU int64; dtype conversion never casts metadata.
These device rules follow upstream
[PackedSequence](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html).

## Recurrent Integration

RNN, GRU and LSTM accept PackedSequence wherever they accept a dense input and
return `(packed_output, final_state)`. Module `batch_first` does not affect
packed data. Initial and final states use original, unsorted batch order and
the regular batched shapes, including projected LSTM hidden width. Output
preserves batch sizes and both permutations.

Each timestep applies the Sputnik cell only to its active batch prefix. The
forward direction retains final states of completed rows. The reverse direction
introduces newly active rows with their initial states. Higher layers consume
packed direction-concatenated outputs, and inter-layer dropout samples only
real packed rows. No padded positions enter recurrent gates or gradients.
The existing bang/RNG and explicit-state contracts are unchanged.

## Verification And Limits

`tests/packed.s` compares packing/padding values and gradients with test-only
ATen CPU primitives. Its recurrent oracle checks packed tanh/ReLU RNN, GRU,
LSTM, one/two layers, both directions, nonzero initial state/biases, bias-free
modules, projected LSTM and every parameter gradient. Other cases cover
unsorted batch restoration, both layouts, left/right/empty-sequence padding,
zero padding gradients, independent dense-sequence equivalence, protected
metadata, transfers, aliases, state round trips, dropout replay/endpoints,
invalid calls and grad/inference scopes.

`tests/packed_mps.s` runs the same Sputnik implementation on actual MPS with
automatic CPU fallback disabled. The shared native test accepts `--device mps`.
CPU oracle transfers are explicit verification only; production sequence
data, cell operations and gradients stay on the selected device. CPU batch
sizes and explicit CPU Generator mask sampling are not MPS computation fallback.

CPU and actual MPS runs pass in both VM and shared full-native execution.
The native graph compiles 2295/2295 code bodies with full body coverage,
zero VM/bytecode fallback and VM-independent execution. Both native device
runs use an empty PATH and audit loaded libraries for absence of Python and
libtorch_python.
Dense recurrent VM regressions pass on CPU/MPS; production NN catalog tests
pass on CPU/MPS, with CPU initialization/container and MPS auto-registration
checks. Production manifests exclude the test oracle sources.

Packed ONNX export, pin-memory helpers, fused recurrent execution/parameter
flattening, Python recurrent state-dict names and identical upstream random
initialization are not claimed. ONNX tracing still rejects Tensor-to-host
metadata extraction instead of silently exporting a data-dependent layout.
