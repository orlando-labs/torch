# Loss Functions And Distances

This block uses LibTorch 2.14.0 ATen kernels, without Python dependencies.
Module configuration, Gaussian/multilabel/custom-distance composition and
reduction policy stay in Sputnik. Each public class has its own source file.
All functions are non-mutating and accept read-only snapshots. Outputs retain
autograd, dtype and device according to the underlying operators.

## API

Import functions from `torch.nn.functional` and classes from
`torch.nn`. Reductions accept Symbols or strings: `none`, `mean`, `sum`.
The default is `mean`. KL divergence additionally accepts `batchmean`:
the sum divided by the first input dimension (a scalar input is not divided).

| Class | Functional Form | Options Besides Reduction |
| --- | --- | --- |
| PoissonNLLLoss | poisson_nll_loss | log_input: true, full: false, eps: 1e-8 |
| GaussianNLLLoss | gaussian_nll_loss | full: false, eps: 1e-6 |
| SoftMarginLoss | soft_margin_loss | none |
| MultiLabelSoftMarginLoss | multilabel_soft_margin_loss | weight: null |
| HingeEmbeddingLoss | hinge_embedding_loss | margin: 1.0 |
| MarginRankingLoss | margin_ranking_loss | margin: 0.0 |
| CosineEmbeddingLoss | cosine_embedding_loss | margin: 0.0 |
| MultiMarginLoss | multi_margin_loss | p: 1, margin: 1.0, weight: null |
| MultiLabelMarginLoss | multilabel_margin_loss | none |
| TripletMarginLoss | triplet_margin_loss | margin: 1.0, p: 2.0, eps: 1e-6, swap: false |
| TripletMarginWithDistanceLoss | triplet_margin_with_distance_loss | distance_function: null, margin: 1.0, swap: false |
| CTCLoss | ctc_loss | blank: 0, zero_infinity: false |

`PairwiseDistance`/`pairwise_distance` accept p: 2.0, eps: 1e-6, keepdim: false.
Epsilon is added to each difference coordinate before taking the norm.
`CosineSimilarity`/`cosine_similarity` accept dim: 1, eps: 1e-8.
These distance operations do not have a reduction option.

Weights on MultiMarginLoss and MultiLabelSoftMarginLoss are registered buffers,
not parameters, and participate in state_dict/load_state_dict!.
MultiLabelSoftMarginLoss uses stable log-sigmoid and averages over the last
class axis before applying the requested sample reduction.

## Gaussian NLL

`gaussian_nll_loss(input, target, var, ...)` accepts Tensor variance or a
numeric scalar. Tensor variance must have the input shape, omit the final
axis, or differ in exactly one axis where variance has size 1. Negative
variance is rejected by reading a scalar check, which synchronizes MPS.
Broadcasting of target follows tensor operations.

Variance is cloned, then clamped to eps under no-grad. The original variance
is unchanged; stability clamping is ignored by differentiation, including
higher-order derivatives. With full: true, the loss includes 0.5 * log(2*pi).
See the [Gaussian NLL contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GaussianNLLLoss.html).

## Custom Triplet Distance

```sputnik
from torch.nn import TripletMarginWithDistanceLoss, PairwiseDistance

def squared_distance(a, b): (a - b).square.sum(dim: -1)

criterion = TripletMarginWithDistanceLoss(distance_function: &squared_distance)
loss = criterion(anchor, positive, negative)
metric = TripletMarginWithDistanceLoss(distance_function: PairwiseDistance(eps: 0))
```

The distance can be a Sputnik function/closure or callable Module and should
return nonnegative per-example distances. Null selects pairwise_distance.
All three inputs must have the same rank, and margin must be positive.
Swap uses the minimum of anchor-negative and positive-negative distances;
ties inherit ATen's gradient split. The final lower clamp inherits the exact
ATen boundary derivative (zero at the boundary in LibTorch 2.14.0).
See the [custom-distance contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TripletMarginWithDistanceLoss.html).

## CTC

`ctc_loss(log_probs, targets, input_lengths, target_lengths, ...)` accepts
log probabilities shaped [T, N, C] or unbatched [T, C]. Targets are integer
Tensor labels, either concatenated [sum(target_lengths)] or padded [N, S].
Lengths can be Array/Tuple values or scalar/1D int32/int64 tensors; their
element counts must match the batch. Target tensors accept int32/int64.

Input lengths must lie in [0, T], target lengths must be nonnegative and fit
the target representation. Active labels must lie in [0, C) and cannot equal
blank. Padded entries outside the active lengths are ignored, so sentinels
such as -1 are allowed there. Blank defaults to 0 and must be in [0, C).

None returns one loss per example (a scalar for unbatched input). Mean divides
each loss by its target length clamped to at least 1, then averages the batch.
Impossible alignments produce infinity; zero_infinity: true replaces them
and their gradients with zero. See the [CTC contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html).

Tensor length metadata is explicitly read on CPU at the boundary. Active
target validation also reads a CPU copy and introduces synchronization for
MPS targets. The actual CTC computation and gradients remain on the input
backend; this is not an automatic CPU computation fallback.

## Verification

`tests/losses.s` and `tests/losses_mps.s` share numerical references and
derivatives, Gaussian clamping/second derivatives, stable extreme logits,
weight state restore, custom callable distances, margin boundary cases,
CTC path enumeration, padded/concatenated targets, mixed length metadata,
different sequence lengths, zero_infinity, invalid inputs and execution scopes.

MPS runs disable automatic CPU fallback and check actual output/gradient
placement. MultiMarginLoss and MultiLabelMarginLoss have no MPS kernels in
the verified build and raise explicitly. The other new functions, including
CTC forward/backward, execute on actual MPS. Float64 is tested on CPU only.

The tests run in VM and full-native modes on CPU and actual MPS. Native builds
require full body coverage and no VM fallback. Executables are also run with
an empty PATH and checked for absence of Python runtime libraries.
