# Normalization And Dropout

Verified against LibTorch 2.14.0, on CPU and MPS with automatic CPU fallback
disabled. Composition, counters, random-state ownership and module registries
stay in Sputnik. Only BatchNorm/InstanceNorm and pooling kernels cross ATen.

## Normalization

`torch.nn` exports BatchNorm1d/2d/3d, InstanceNorm1d/2d/3d and
LocalResponseNorm, alongside the existing LayerNorm, RMSNorm and GroupNorm.

BatchNorm and InstanceNorm accept `features`, `eps:`, `momentum:`, `affine:`,
`track_running_stats:`, `device:`, `dtype:` and `bias:`. Default dtype is
float32. `bias: false` omits the affine bias while retaining the scale.
BatchNorm defaults to affine/tracking enabled; InstanceNorm disables both.

```sputnik
from torch.nn import BatchNorm2d, InstanceNorm2d

batch = BatchNorm2d(32, momentum: null, device: "mps")
y = batch.call!(images)
batch.eval!
z = batch(images)

instance = InstanceNorm2d(32, affine: true, device: "mps")
y = instance(images)
```

Training with tracked buffers requires `call!`/`forward!`. These calls do not
mutate the input. Evaluation or untracked normalization uses ordinary calls.
Functional forms are `nn.functional.batch_norm`/`batch_norm!` and
`instance_norm`/`instance_norm!`; bang forms permit tracked-stat updates.
Non-bang functional calls reject input-stat normalization with running buffers.

BatchNorm `momentum: null` uses `1 / (num_batches_tracked + 1)` for each
successful batch. The counter survives state-dict restore. Fixed momentum
keeps exponential moving averages. Normalized output uses biased variance;
running variance uses the unbiased estimator. A training batch with only
one value per channel is rejected before changing buffers or the counter.
Reading a cumulative counter on MPS introduces synchronization.

InstanceNorm accepts `[N, C, *spatial]` or unbatched `[C, *spatial]` for its
dimensionality. With no tracking, train and eval both use each instance's
spatial statistics. Tracked eval uses running statistics. A single spatial
element is rejected when using input statistics. Channels must match `features` when
affine parameters or running buffers are present; without either, features
does not constrain the input channel count.

Tracked InstanceNorm registers running_mean/running_var and the conventional
num_batches_tracked buffer, which is initialized to zero and not incremented.
Unlike BatchNorm, InstanceNorm
does not implement cumulative averaging: module `momentum: null` means a
zero update factor. This follows the
[upstream InstanceNorm implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/modules/instancenorm.py).

Both families expose `reset_running_stats!` and `reset_parameters!`. Reset
sets mean/counter to zero, variance/affine scale to one and bias to zero,
preserving parameter identity and leaf/autograd properties. State dictionaries
contain registered tensors and use the existing strict load contract.

`LocalResponseNorm(size, alpha: 1e-4, beta: 0.75, k: 1.0)` and
`nn.functional.local_response_norm` normalize across channels, including
even-sized windows. They accept rank >= 3 and return the input unchanged for
empty tensors. Sputnik uses square, padding, average pooling and power to
implement the [LRN formula](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LocalResponseNorm.html).

## Dropout

All six modules own an explicit Generator, accept `p: 0.5, seed: 0`, and
inherit `rng_state`/`load_rng_state!`. Random training calls require `call!`;
eval, p=0 and p=1 do not consume RNG. Outputs never mutate their inputs.

| Module / Function | Noise Shape |
| --- | --- |
| Dropout / dropout | every input element |
| Dropout1d / dropout1d | one mask per channel; rank 2 or 3 |
| Dropout2d / dropout2d | one mask per channel; rank 3 or 4 |
| Dropout3d / dropout3d | one mask per channel; rank 4 or 5 |
| AlphaDropout / alpha_dropout | every input element |
| FeatureAlphaDropout / feature_alpha_dropout | one mask per `[N, C]` channel; rank >= 2 |

Channel Dropout accepts batched or unbatched inputs. In particular,
Dropout2d treats rank 3 as `[C, H, W]`, matching the installed
[LibTorch C++ frontend](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/api/include/torch/nn/functional/dropout.h),
not Python's historical rank-3 interpretation as `[N, C, L]`.

Regular Dropout rescales retained values by `1 / (1-p)`. Alpha variants use
SELU's negative saturation value and affine rescaling to preserve zero mean
and unit variance. p=1 returns input times zero for all variants. Stochastic
inputs must be floating tensors; backend support follows ATen.

Pure function names preserve non-mutating semantics. Random training requires
an explicit boolean `mask:` of the noise shape, on the input device:

```sputnik
import torch.nn.functional as f
from torch.random.generator import Generator

y = f.dropout(x, mask: keep)
z = f.dropout!(x, Generator(seed: 7))
```

Every function has a bang form taking a Generator as its second argument.
The bang denotes RNG-state mutation, not input mutation. Functional ordinary
Dropout defaults to training; alpha functions default to evaluation, matching
Torch. Eval/p=0/empty inputs are identity and retain read-only/alias status.
Invalid configuration and ONNX random training are rejected before consuming
RNG. In-place input Dropout is not exposed.

The existing Generator samples float32 uniforms on CPU and explicitly
transfers masks to MPS. This is not automatic backend fallback. Replay is
defined by saved Generator state and the same sampling configuration; no
bit-identical sequence with Torch's global Bernoulli RNG is promised.

## Tests

`tests/normalization.s` and `tests/normalization_mps.s` share the same checks:
numerical output and running-stat references, state restore/reset, dtype/bias,
unbatched and higher-rank layouts, closed-form LRN/dropout derivatives,
channel coherence, RNG replay/no-op preservation, read-only preflight and
ONNX mutation rejection. They run in VM and full-native modes. MPS assertions
check result/gradient placement, with CPU fallback disabled.
