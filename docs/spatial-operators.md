# Spatial Operators

The API uses LibTorch 2.14.0 ATen kernels, without Python dependencies.
Validation, unpool output-shape policy and LP pooling composition stay in
Sputnik; interpolation scale-to-size lowering occurs at the ATen boundary.
All new operations are non-mutating; results own storage, while inputs and
read-only snapshots are preserved. Autograd follows the underlying operators.

## Interpolation

`torch.nn.functional.interpolate` accepts `[N, C, *spatial]` with one,
two or three spatial axes:

```sputnik
import torch.nn.functional as f
from torch.nn import Upsample

y = f.interpolate(x, size: [64, 96], mode: :bilinear, align_corners: false)
z = f.interpolate(x, scale_factor: 1.5, mode: :bicubic, antialias: true)
layer = Upsample(size: [64, 96], mode: :bilinear)
y = layer(x)
```

Exactly one of `size:` and `scale_factor:` is required. A scalar is repeated
across spatial axes; Array/Tuple lengths must match the spatial rank.
Output lengths are positive Ints. Scales are finite positive numbers; output
sizes use floor(input_size * scale). Empty or overflowing scaled outputs
are rejected before dispatch.

| Mode | Spatial Rank | Options |
| --- | --- | --- |
| nearest | 1, 2, 3 | legacy floor coordinates |
| nearest-exact | 1, 2, 3 | center-based nearest coordinates |
| linear | 1 | align_corners |
| bilinear | 2 | align_corners; antialias |
| bicubic | 2 | align_corners; antialias |
| trilinear | 3 | align_corners |
| area | 1, 2, 3 | adaptive-average pooling |
| lanczos | 2, CPU only | requires antialias: true |

Use the string `"nearest-exact"` for its hyphenated mode name. Both strings
and Symbols are accepted for other modes. Default mode is nearest.
`align_corners: null` resolves to false for interpolating kernels; specifying
it for nearest/nearest-exact/area is an error. Lanczos forwards its alignment
flag to the ATen kernel.

`recompute_scale_factor: true` derives sampling scales from the rounded
output shape, rather than using the supplied factors directly. It requires
scale_factor, not size. Area always uses the computed shape. Antialias is
available only for 2D bilinear/bicubic/Lanczos; it defaults to false.
Bicubic/Lanczos may overshoot; no output clamp is added.

`Upsample` exposes the same keyword options as interpolate. The convenience
modules `UpsamplingNearest2d` and `UpsamplingBilinear2d` accept size or scale;
the legacy bilinear module uses align_corners: true. These modules have no
parameters, buffers or stochastic state.

Coordinate and scale semantics follow the
[Torch interpolation contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.interpolate.html).
Supported dtypes and devices are determined by ATen. Verified MPS limitations:
Lanczos is CPU-only, and area/adaptive pooling requires input spatial lengths
divisible by output lengths. Unsupported cases raise, without enabling CPU
fallback or silently transferring the input.

## Pooling Indices

MaxPool1d/2d/3d and AdaptiveMaxPool1d/2d/3d now accept
`return_indices: false`. With true, the result is a Tuple `(values, indices)`;
otherwise it remains a Tensor. The same option is available on their
`nn.functional.max_pool1d/2d/3d` and `adaptive_max_pool1d/2d/3d` forms.
Max pooling retains kernel_size, stride, padding, dilation and ceil_mode.
Adaptive max accepts null entries in its output-size Array/Tuple to retain
the corresponding spatial axis.

Indices are int64 tensors on the input device, with no gradients. Each index
is flattened within one channel's spatial plane/volume; batch/channel
offsets are not included. Values remain differentiable. In the verified MPS
build, adaptive MaxPool3d is unavailable; adaptive 1D/2D also has the
divisibility restriction above. Ordinary MaxPool3d is supported and tested.

## Unpooling

```sputnik
from torch.nn import MaxPool2d, MaxUnpool2d

(values, indices) = MaxPool2d(2, return_indices: true)(image)
restored = MaxUnpool2d(2)(values, indices, output_size: image.shape)
```

MaxUnpool1d/2d/3d and their functional forms accept pooled values, indices,
kernel_size, optional stride/padding, and `output_size: null`. Modules store
kernel/stride/padding and accept output_size on each forward call.
Inputs may be batched or unbatched. Indices must match the input's shape,
device and int64 dtype. Read-only inputs are accepted, and outputs are
independent writable tensors.

Default spatial shape is `(pooled_length - 1) * stride + kernel - 2 * padding`.
Explicit output_size is either the spatial Array/Tuple or `[N, C, *spatial]`;
only its spatial part is used. Each requested length must lie strictly
between default_length - stride and default_length + stride, resolving
pooling-size ambiguity. Output lengths must be positive.

Unpool places values at saved indices and fills remaining positions with
zero. It is only a partial inverse: discarded non-maximal values cannot be
recovered. Duplicate indices retain ATen's nondeterministic overwrite
limitations; no accumulation semantics are added. See the
[MaxUnpool contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxUnpool2d.html).

MPS bounds preflight reduces indices on their own device and reads scalar
bounds before dispatch, introducing synchronization. Invalid indices raise
a rescuable error instead of poisoning the Metal command stream. Normal
1D/2D/3D unpool computations stay on MPS, with CPU fallback disabled.

## LP Pooling

LPPool1d/2d/3d and `nn.functional.lp_pool1d/2d/3d` accept positive norm_type,
kernel_size, optional stride (default kernel_size), and ceil_mode: false.
They use signed powers, not abs(input) powers, matching LibTorch's LP
pooling composition. Negative inputs with fractional powers may produce
NaNs according to ATen's power semantics.

Sputnik composes power, average pooling, sign, absolute/ReLU, window-volume
scaling and inverse power. In particular, ceil-mode partial windows retain
LibTorch's average-pool/window-volume convention. Zero-sum windows have the
same zero-gradient behavior as the upstream composition. No padding option
is exposed. The
[LibTorch functional implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/api/include/torch/nn/functional/pooling.h)
is the reference for these semantics.

## Verification

`tests/spatial.s` and `tests/spatial_mps.s` share numerical references,
fractional/recomputed scales, antialias filters, noncontiguous/snapshot
inputs, all supported interpolation modes, tuple indices, unpool size
ambiguity, LP values/derivatives, invalid inputs and explicit backend limits.
They run in VM and full-native modes on CPU and actual MPS. The MPS runs
disable automatic CPU fallback and check output/index/gradient placement.
Full-native executables are also run without Python on PATH or in loaded
runtime libraries.
