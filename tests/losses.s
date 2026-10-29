package torch.losses_test

from torch import tensor, zeros, ones, full
from torch.nn import PoissonNLLLoss, GaussianNLLLoss, SoftMarginLoss, MultiLabelSoftMarginLoss
from torch.nn import HingeEmbeddingLoss, MarginRankingLoss, CosineEmbeddingLoss, MultiMarginLoss, MultiLabelMarginLoss
from torch.nn import TripletMarginLoss, TripletMarginWithDistanceLoss, CTCLoss, PairwiseDistance, CosineSimilarity, KLDivLoss
from torch.autograd import no_grad, inference_mode, grad
from torch.checks import check
import torch.nn.functional as f

export main, run

def close(x, expected, message, tolerance: 3e-4):
  check(x.shape == expected.shape and (x - expected).abs.sum.item < tolerance, message)

def rejected(&block):
  failed = false
  try:
    block()
  rescue ValueError:
    failed = true
  rescue ArgumentError:
    failed = true
  rescue Torch.Errors.Error:
    failed = true
  check(failed, "invalid loss must raise")

def squared_distance(a, b): (a - b).square.sum(dim: -1)

def run(device):
  log2 = Math.log(2.0)
  log4 = Math.log(4.0)
  x = tensor([0.0, log2], device:, requires_grad: true)
  target = tensor([0.0, 2.0], device:)
  close(PoissonNLLLoss(reduction: :none)(x, target), tensor([1.0, 2 - 2 * log2], device:), "Poisson log-input reference")
  f.poisson_nll_loss(x, target, reduction: :sum).backward!
  close(x.grad, tensor([1.0, 0.0], device:), "Poisson log-input derivative")
  rate = tensor([1.0, 2.0], device:, requires_grad: true)
  f.poisson_nll_loss(rate, target, log_input: false, eps: 0, reduction: :sum).backward!
  close(rate.grad, tensor([1.0, 0.0], device:), "Poisson rate derivative")
  correction = 2 * log2 - 2 + 0.5 * Math.log(4 * Math.PI)
  close(f.poisson_nll_loss(x, target, full: true, reduction: :none),
    tensor([1.0, 2 - 2 * log2 + correction], device:), "Poisson Stirling term")

  mean = tensor([1.0, 3.0], device:, requires_grad: true)
  variance = tensor([1.0, 4.0], device:, requires_grad: true)
  observations = zeros([2], device:)
  expected = tensor([0.5, 0.5 * (log4 + 2.25)], device:)
  close(GaussianNLLLoss(reduction: :none)(mean, observations, variance), expected, "Gaussian NLL reference")
  f.gaussian_nll_loss(mean, observations, variance, reduction: :sum).backward!
  close(mean.grad, tensor([1.0, 0.75], device:), "Gaussian mean derivative")
  close(variance.grad, tensor([0.0, -0.15625], device:), "Gaussian variance derivative")
  close(f.gaussian_nll_loss(mean, observations, variance, full: true, reduction: :none),
    expected + 0.5 * Math.log(2 * Math.PI), "Gaussian full constant")
  clamped = tensor([0.1], device:, requires_grad: true)
  f.gaussian_nll_loss(tensor([1.0], device:), zeros([1], device:), clamped, eps: 0.5, reduction: :sum).backward!
  close(clamped.grad, tensor([-1.0], device:), "variance derivative ignores stability clamp")
  close(clamped, tensor([0.1], device:), "Gaussian does not mutate variance")
  second_var = tensor([0.1], device:, requires_grad: true)
  second_loss = f.gaussian_nll_loss(ones([1], device:), zeros([1], device:), second_var, eps: 0.5)
  derivative = grad([second_loss], [second_var], create_graph: true)[0]
  close(grad([derivative.sum], [second_var])[0], tensor([6.0], device:), "Gaussian second variance derivative")
  close(f.gaussian_nll_loss(ones([2, 3], device:), zeros([2, 3], device:), ones([2], device:), reduction: :none),
    full([2, 3], 0.5, device:), "Gaussian one-short variance")
  close(f.gaussian_nll_loss(ones([2, 3], device:), zeros([2, 3], device:), ones([1, 3], device:), reduction: :none),
    full([2, 3], 0.5, device:), "Gaussian singleton axis variance")
  close(f.gaussian_nll_loss(ones([2, 3], device:), zeros([2, 3], device:), 1.0), tensor(0.5, device:), "Gaussian scalar variance")
  close(f.gaussian_nll_loss(mean.snapshot, observations.snapshot, variance.snapshot, reduction: :none), expected,
    "Gaussian accepts snapshots")
  rejected(): f.gaussian_nll_loss(mean, observations, tensor([-1.0, 1.0], device:))
  rejected(): f.gaussian_nll_loss(ones([2, 3], device:), zeros([2, 3], device:), ones([1, 1], device:))
  rejected(): f.gaussian_nll_loss(mean, observations, variance, reduction: :invalid)

  logits = zeros([2], device:, requires_grad: true)
  labels = tensor([1.0, -1.0], device:)
  close(SoftMarginLoss(reduction: :none)(logits, labels), full([2], log2, device:), "soft-margin reference")
  f.soft_margin_loss(logits, labels, reduction: :sum).backward!
  close(logits.grad, tensor([-0.5, 0.5], device:), "soft-margin derivative")
  scores = zeros([2, 2], device:, requires_grad: true)
  binary_targets = tensor([[1.0, 0.0], [0.0, 1.0]], device:)
  weights = tensor([2.0, 1.0], device:)
  criterion = MultiLabelSoftMarginLoss(weight: weights, reduction: :none)
  close(criterion(scores, binary_targets), full([2], 1.5 * log2, device:), "multilabel soft-margin class averaging")
  check(criterion.named_buffers.has_key?("weight") and criterion.parameters.empty?, "loss weight is a buffer")
  restored = MultiLabelSoftMarginLoss(weight: ones([2], device:), reduction: :none)
  saved = criterion.state_dict
  check(saved["weight"].readonly? and not criterion.named_buffers["weight"].readonly?, "state dict snapshots preserve live buffers")
  restored.load_state_dict!(saved)
  close(restored(scores, binary_targets), criterion(scores, binary_targets), "loss weight state restore")
  f.multilabel_soft_margin_loss(scores, binary_targets, weight: weights, reduction: :sum).backward!
  close(scores.grad, tensor([[-0.5, 0.25], [0.5, -0.25]], device:), "multilabel soft-margin derivative")
  close(f.multilabel_soft_margin_loss(zeros([1, 2], device:).fill!(1000), tensor([[1.0, 0.0]], device:)),
    tensor(500.0, device:), "stable multilabel log-sigmoid")

  distances = tensor([0.2, 0.3, 2.0], device:, requires_grad: true)
  signed = tensor([1.0, -1.0, -1.0], device:)
  close(HingeEmbeddingLoss(reduction: :none)(distances, signed), tensor([0.2, 0.7, 0.0], device:), "hinge embedding")
  f.hinge_embedding_loss(distances, signed, reduction: :sum).backward!
  close(distances.grad, tensor([1.0, -1.0, 0.0], device:), "hinge derivative")
  a = tensor([1.0, 1.0], device:, requires_grad: true)
  b = tensor([2.0, 0.0], device:, requires_grad: true)
  close(MarginRankingLoss(margin: 0.5, reduction: :none)(a, b, labels), tensor([1.5, 1.5], device:), "ranking reference")
  f.margin_ranking_loss(a, b, labels, margin: 0.5, reduction: :sum).backward!
  close(a.grad, tensor([-1.0, 1.0], device:), "ranking first derivative")
  close(b.grad, tensor([1.0, -1.0], device:), "ranking second derivative")
  vectors1 = tensor([[1.0, 0.0], [1.0, 0.0]], device:, requires_grad: true)
  vectors2 = tensor([[0.0, 1.0], [1.0, 1.0]], device:, requires_grad: true)
  expected = tensor([1.0, 1.0 / (2.0 ** 0.5) - 0.2], device:)
  close(CosineEmbeddingLoss(margin: 0.2, reduction: :none)(vectors1, vectors2, labels), expected, "cosine embedding reference")
  f.cosine_embedding_loss(vectors1, vectors2, labels, margin: 0.2).backward!
  check(vectors1.grad != null and vectors2.grad != null, "cosine embedding gradients")
  close(CosineSimilarity()(vectors1, vectors2), tensor([0.0, 1.0 / (2.0 ** 0.5)], device:), "cosine similarity module")
  close(f.cosine_similarity(zeros([2, 2], device:), vectors2), zeros([2], device:), "cosine zero-vector stability")
  pair_input = tensor([[3.0, 4.0]], device:, requires_grad: true)
  close(PairwiseDistance(eps: 0, keepdim: true)(pair_input, zeros([1, 2], device:)), tensor([[5.0]], device:), "pairwise distance module")
  f.pairwise_distance(pair_input, zeros([1, 2], device:), eps: 0).sum.backward!
  close(pair_input.grad, tensor([[0.6, 0.8]], device:), "pairwise distance derivative")
  close(f.pairwise_distance(zeros([2], device:), zeros([2], device:), eps: 0.1),
    tensor(0.1 * (2.0 ** 0.5), device:), "pairwise epsilon is per-coordinate")

  anchor = tensor([[0.0, 0.0]], device:, requires_grad: true)
  positive = tensor([[1.0, 0.0]], device:, requires_grad: true)
  negative = tensor([[1.5, 0.0]], device:, requires_grad: true)
  close(TripletMarginLoss(eps: 0, reduction: :none)(anchor, positive, negative), tensor([0.5], device:), "triplet reference")
  f.triplet_margin_loss(anchor, positive, negative, eps: 0, reduction: :sum).backward!
  close(anchor.grad, zeros([1, 2], device:), "triplet anchor derivative")
  close(positive.grad, tensor([[1.0, 0.0]], device:), "triplet positive derivative")
  close(negative.grad, tensor([[-1.0, 0.0]], device:), "triplet negative derivative")
  close(f.triplet_margin_loss(anchor, positive, negative, eps: 0, swap: true), tensor(1.5, device:), "triplet distance swap")
  custom = TripletMarginWithDistanceLoss(distance_function: &squared_distance, margin: 2, reduction: :none)
  close(custom(anchor, positive, negative), tensor([0.75], device:), "Sputnik callable distance")
  close(f.triplet_margin_with_distance_loss(anchor, positive, negative, distance_function: &squared_distance,
    margin: 2, swap: true), tensor(2.75, device:), "custom-distance swap")
  module_distance = TripletMarginWithDistanceLoss(distance_function: PairwiseDistance(eps: 0))
  close(module_distance(anchor, positive, negative), tensor(0.5, device:), "Module callable distance")
  close(TripletMarginWithDistanceLoss()(anchor, positive, negative),
    f.triplet_margin_loss(anchor, positive, negative), "default distance parity")
  pa = tensor([[0.0, 0.0]], device:, requires_grad: true)
  pp = tensor([[1.0, 0.0]], device:, requires_grad: true)
  pn = tensor([[2.0, 0.0]], device:, requires_grad: true)
  f.triplet_margin_with_distance_loss(pa, pp, pn, distance_function: PairwiseDistance(eps: 0), reduction: :sum).backward!
  close(pp.grad, zeros([1, 2], device:), "zero-loss boundary uses LibTorch clamp derivative")
  close(pn.grad, zeros([1, 2], device:), "zero-loss boundary negative derivative")
  rejected(): TripletMarginLoss(margin: 0)
  rejected(): TripletMarginWithDistanceLoss(margin: -1)
  rejected(): f.triplet_margin_with_distance_loss(anchor, positive[0], negative)

  ranking_scores = tensor([[0.2, 0.5, 0.1]], device:, requires_grad: true)
  rank_target = tensor([1], dtype: "int64", device:)
  multi = MultiMarginLoss(weight: tensor([1.0, 2.0, 3.0], device:), reduction: :none)
  if device == "cpu":
    close(multi(ranking_scores, rank_target), tensor([2.6 / 3], device:), "weighted multi-margin p1")
    close(f.multi_margin_loss(ranking_scores, rank_target, p: 2, reduction: :none), tensor([0.85 / 3], device:), "multi-margin p2")
    f.multi_margin_loss(ranking_scores, rank_target, reduction: :sum).backward!
    close(ranking_scores.grad, tensor([[1.0 / 3, -2.0 / 3, 1.0 / 3]], device:), "multi-margin derivative")
    ml_scores = tensor([[0.5, 0.2, 0.1]], device:, requires_grad: true)
    ml_target = tensor([[0, 2, -1]], dtype: "int64", device:)
    close(MultiLabelMarginLoss(reduction: :none)(ml_scores, ml_target), tensor([1.8 / 3], device:), "multilabel margin reference")
    f.multilabel_margin_loss(ml_scores, ml_target, reduction: :sum).backward!
    close(ml_scores.grad, tensor([[-1.0 / 3, 2.0 / 3, -1.0 / 3]], device:), "multilabel margin derivative")
  else:
    rejected(): multi(ranking_scores, rank_target)
    rejected(): MultiLabelMarginLoss()(ranking_scores, tensor([[0, 2, -1]], dtype: "int64", device:))
  check(multi.state_dict.has_key?("weight"), "multi-margin weight state")
  rejected(): MultiMarginLoss(p: 3)
  rejected(): f.multi_margin_loss(ranking_scores, tensor([3], dtype: "int64", device:))

  log_probs = tensor([[[Math.log(0.6), Math.log(0.4)]], [[Math.log(0.6), Math.log(0.4)]]], device:)
  ctc_target = tensor([1], dtype: "int64", device:)
  ctc = CTCLoss(reduction: :none)
  expected_ctc = -Math.log(0.64)
  close(ctc(log_probs, ctc_target, [2], [1]), tensor([expected_ctc], device:), "CTC enumerated-path reference")
  lengths = tensor([2], dtype: "int32", device:)
  target_lengths = tensor([1], dtype: "int64", device:)
  close(f.ctc_loss(log_probs, tensor([[1, -1]], dtype: "int64", device:), lengths, target_lengths),
    tensor(expected_ctc, device:), "CTC padded targets and Tensor lengths")
  close(f.ctc_loss(log_probs, ctc_target, (2,), target_lengths), tensor(expected_ctc, device:), "CTC mixed length metadata")
  batched_probs = tensor([[[Math.log(0.6), Math.log(0.4)], [Math.log(0.6), Math.log(0.4)]],
    [[Math.log(0.6), Math.log(0.4)], [Math.log(0.6), Math.log(0.4)]],
    [[Math.log(0.6), Math.log(0.4)], [Math.log(0.6), Math.log(0.4)]]], device:)
  batched_targets = tensor([[1, -1], [1, 1]], dtype: "int32", device:)
  batched_expected = tensor([-Math.log(0.64), -Math.log(0.096)], device:)
  close(f.ctc_loss(batched_probs, batched_targets, [2, 3], [1, 2], reduction: :none),
    batched_expected, "CTC different input and target lengths")
  close(f.ctc_loss(batched_probs, batched_targets, [2, 3], [1, 2], reduction: :mean),
    tensor((expected_ctc - Math.log(0.096) / 2) / 2, device:), "CTC mean normalizes target lengths")
  close(f.ctc_loss(log_probs.squeeze(dim: 1), ctc_target, [2], [1], reduction: :none), tensor(expected_ctc, device:), "unbatched CTC")
  ctc_logits = log_probs.copy.requires_grad!(true)
  CTCLoss(reduction: :sum)(ctc_logits.log_softmax(dim: -1), ctc_target, [2], [1]).backward!
  close(ctc_logits.grad, tensor([[[0.225, -0.225]], [[0.225, -0.225]]], device:), "CTC log-softmax backward")
  repeated = tensor([1, 1], dtype: "int64", device:)
  impossible = log_probs.copy.requires_grad!(true)
  close(CTCLoss(zero_infinity: true)(impossible.log_softmax(dim: -1), repeated, [2], [2]), tensor(0.0, device:), "CTC zero infinity")
  f.ctc_loss(impossible.log_softmax(dim: -1), repeated, [2], [2], zero_infinity: true).backward!
  close(impossible.grad, zeros(impossible.shape, device:), "CTC zero-infinity gradient")
  close(f.ctc_loss(log_probs, zeros([0], dtype: "int64", device:), [2], [0]), tensor(-Math.log(0.36), device:), "CTC empty target")
  swapped = log_probs.index([:all, :all, tensor([1, 0], dtype: "int64", device:)])
  close(f.ctc_loss(swapped, tensor([0], dtype: "int64", device:), [2], [1], blank: 1), tensor(expected_ctc, device:), "CTC nonzero blank")
  rejected(): ctc(log_probs, ctc_target, [3], [1])
  rejected(): ctc(log_probs, ctc_target, [2], [-1])
  rejected(): ctc(log_probs, ctc_target, [2, 2], [1])
  rejected(): ctc(log_probs, ctc_target, [2], [2])
  rejected(): ctc(log_probs, tensor([2], dtype: "int64", device:), [2], [1])
  rejected(): ctc(log_probs, tensor([0], dtype: "int64", device:), [2], [1])
  rejected(): ctc(log_probs, ctc_target.to(device:, dtype: "float32"), [2], [1])
  rejected(): f.ctc_loss(log_probs, ctc_target, [2], [1], blank: 2)
  close(ctc(log_probs.snapshot, ctc_target.snapshot, [2], [1]), tensor([expected_ctc], device:), "CTC snapshots after invalid calls")
  check(ctc_logits.grad.device.type == device and ctc(log_probs, ctc_target, [2], [1]).device.type == device,
        "CTC actual backend")

  distribution = tensor([[0.25, 0.75], [0.5, 0.5]], device:)
  prediction = full([2, 2], 0.5, device:).log
  reference = (distribution * (distribution.log - prediction)).sum / 2
  close(KLDivLoss(reduction: :batchmean)(prediction, distribution), reference, "KL batchmean")
  close(f.kl_div(prediction, distribution.log, log_target: true, reduction: :batchmean), reference, "KL log-target batchmean")
  detached_loss = no_grad(): f.gaussian_nll_loss(mean, observations, variance)
  detached_distance = inference_mode(): f.pairwise_distance(pair_input, zeros([1, 2], device:))
  check(not detached_loss.requires_grad?, "loss no-grad scope")
  check(not detached_distance.requires_grad?, "distance inference scope")
  check(mean.grad.device.type == device and scores.grad.device.type == device and positive.grad.device.type == device,
        "loss gradient backends")
  if device == "cpu":
    double = mean.detach.to(device:, dtype: "float64")
    check(f.gaussian_nll_loss(double, double * 0, 1.0).dtype.to_str == "float64", "Gaussian dtype")
  print "PASS torch losses " + device
  0

def main(): run("cpu")
