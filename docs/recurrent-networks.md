# Recurrent Networks

RNN, GRU and LSTM use Sputnik cell composition and sequence traversal over
LibTorch 2.14.0 tensor primitives. There is no opaque C++ recurrent algorithm
in the production adapter and no Python dependency. The tests independently
compare this implementation with ATen's CPU recurrent kernels.

## Sequence API

All three classes accept `input_size`, `hidden_size` and these keywords:

- `num_layers: 1`, `bias: true`, `batch_first: true`, `bidirectional: false`.
- `dropout: 0.0`, `seed: 0`, `device: "cpu"`, `dtype: "float32"`.

RNN additionally accepts `nonlinearity: :tanh` or `:relu` (strings also work).
LSTM additionally accepts `proj_size: 0`. A nonzero projection must be a
positive integer smaller than hidden_size. Sizes and num_layers must be
positive integers. Sequence length must be positive.

SputnikTorch retains its existing batch-first default. To use the upstream
time-first layout, explicitly set `batch_first: false`. A batched input is
[N, T, I] or [T, N, I] respectively. An unbatched input is [T, I], and ignores
batch_first. Noncontiguous input and read-only snapshots are accepted.

Call with `model(input, state: null)`; the result is `(output, final_state)`.
Initial state defaults to zeros. RNN/GRU state is a hidden Tensor; LSTM state
is a `(hidden, memory)` pair. States are explicit values and are never kept
between independent calls.

Let D be 2 for bidirectional, otherwise 1; L be num_layers; H be hidden_size;
P be proj_size when nonzero, otherwise H. Batched output is [N, T, D*P] or
[T, N, D*P]. Hidden state is [L*D, N, P], and LSTM memory is [L*D, N, H].
Unbatched output omits N, and states are [L*D, P] / [L*D, H].

Cells are stored in layer-major order: layer 0 forward, layer 0 reverse,
layer 1 forward, layer 1 reverse, and so on. Reverse outputs are realigned
to original time order before concatenation. A reverse final hidden state
corresponds to original time zero, not the last output position. Higher layers
receive the concatenation of both directions.

All three sequence modules also accept `nn.utils.rnn.PackedSequence` and return
a packed output. Initial/final states remain in original batch order, while
cell traversal follows active packed prefixes in both directions. `batch_first`
does not affect packed inputs. See [Packed sequences](packed-sequences.md) for
utilities, permutation/device invariants and packed gradient coverage.

These shapes, direction ordering and gate formulas follow the upstream
[RNN](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RNN.html),
[GRU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html) and
[LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html) contracts.
GRU uses PyTorch's reset-gate placement after the hidden candidate projection.

## Cells And Projections

RNNCell, GRUCell and LSTMCell accept `bias: true` and `dtype: "float32"` in
addition to seed/device. Both input and recurrent bias parameters disappear
with bias: false. Cells accept [I] or [N, I] inputs and matching states.

LSTMCell also exposes proj_size as a Sputnik extension; upstream provides
this option on the sequence LSTM rather than LSTMCell. Projected hidden state
has size P, while memory and gates retain size H. The hidden-to-gates matrix
uses P input features, and a registered bias-free output projection maps H
to P. All projection weights remain leaf parameters and receive gradients.

Linear now accepts dtype so cell projection matrices and biases have the
requested dtype from construction. Existing Gaussian weight initialization
and zero biases are preserved; matching a PyTorch random seed does not promise
matching initial weights. Numerical comparisons use identical actual weights.
Registry/state-dict names remain Sputnik projection-submodule names rather
than upstream flattened recurrent names.

## Inter-Layer Dropout

Dropout acts on outputs between recurrent layers, after direction concatenation,
never on the last layer or the returned hidden/memory states. It is inactive
in evaluation mode and in single-layer models. No-grad does not imply eval.

Stochastic training with multiple layers and 0 < dropout < 1 requires
`call!` or `forward!`; ordinary call/forward raises before consuming RNG.
The deterministic dropout: 0 and dropout: 1 endpoints allow pure calls.
Invalid input/state shape, dtype or device is rejected before dropout sampling.

`rng_state` and `load_rng_state!` snapshot and restore the explicit Generator.
The existing Generator samples on CPU and explicitly transfers masks to the
input device. Recurrent matrix/gate operations and gradients stay on the
input device; this is not an automatic MPS computation fallback. Parameter
state_dict does not contain RNG state, which is restored separately.

## Verification And Limits

`tests/recurrent.s` and `tests/recurrent_mps.s` compare output, final hidden/
memory, input gradients, initial-state gradients and every parameter gradient
with ATen using the same weights, nonzero biases and initial states. Cases
include tanh/ReLU RNN, GRU, LSTM, one/two layers, both layouts, both directions,
bias-free modules and LSTM projections. Float64 is checked on CPU.

Additional checks cover unbatched inputs/continuation, direction-state ordering,
noncontiguous read-only inputs, state-dict round trips, dropout endpoints,
last-layer exclusion, RNG replay, eval and single-layer no-consumption,
Sequential preflight, invalid calls and grad/inference scopes.

The CPU oracle is linked only into these test entry points, not production
adapter builds. Its CPU conversion is explicit reference verification. MPS
tests disable automatic CPU fallback and check model output and parameter
gradient placement on actual MPS. VM and shared full-native device runs are
required; native tests additionally run with an empty PATH and audit loaded
libraries for absence of a Python runtime.

The original dense-block shared native test compiled all 2056 graph code bodies with full
body coverage, zero VM/native fallback and VM-independent execution. Both CPU
and actual MPS device runs pass. Parser and VM tests plus a full-native corpus
fixture cover tuple assignment after a multiline conditional; this parser
correction was required by the cell-state preparation code.

Packed integration is additionally verified in VM and a shared full-native
test on CPU and actual MPS: 2295/2295 code bodies, zero VM/bytecode fallback,
VM-independent execution and no loaded Python runtime. Its dense-sequence
comparisons exercise both input paths in the same native binary.

Parameter flattening/fused recurrent execution, compatibility
with Python recurrent state-dict keys and exact upstream random initialization
remain outside this block.
