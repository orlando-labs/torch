package torch.embeddings_test

from torch import tensor, zeros, ones
from torch.nn import Embedding, EmbeddingBag
from torch.optim import SGD
from torch.autograd import no_grad, inference_mode
from torch.checks import check
import torch.nn.functional as f

export main, run

def close(x, expected, message):
  check(x.shape == expected.shape and (x - expected).abs.sum.item < 3e-4, message)

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
  check(failed, "invalid embedding must raise")

def run(device):
  initial = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [9.0, 10.0]], device:)
  indices = tensor([0, 1, 1, 2], device:, dtype: "int64")
  offsets = tensor([0, 2, 2, 4], device:, dtype: "int64")
  sum_expected = tensor([[4.0, 6.0], [0.0, 0.0], [8.0, 10.0], [0.0, 0.0]], device:)
  mean_expected = tensor([[2.0, 3.0], [0.0, 0.0], [4.0, 5.0], [0.0, 0.0]], device:)
  max_expected = tensor([[3.0, 4.0], [0.0, 0.0], [5.0, 6.0], [0.0, 0.0]], device:)
  sum_layer = EmbeddingBag(4, 2, weight: initial, mode: :sum)
  close(sum_layer(indices, offsets), sum_expected, "sum bags with empty bags")
  close(EmbeddingBag.from_pretrained(initial)(indices, offsets), mean_expected, "mean pretrained bags")
  max_layer = EmbeddingBag.from_pretrained(initial, freeze: false, mode: :max)
  close(max_layer(indices, offsets), max_expected, "max bags")
  sum_layer(indices, offsets).sum.backward!
  close(sum_layer.weight.grad, tensor([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [0.0, 0.0]], device:), "sum bag gradients")
  max_layer(indices, offsets).sum.backward!
  close(max_layer.weight.grad, tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], device:), "max bag gradients")
  mean_layer = EmbeddingBag(4, 2, weight: initial)
  mean_layer(indices, offsets).sum.backward!
  close(mean_layer.weight.grad, tensor([[0.5, 0.5], [1.0, 1.0], [0.5, 0.5], [0.0, 0.0]], device:), "mean bag gradients")
  padded = EmbeddingBag(4, 2, weight: initial, padding_idx: 1)
  close(padded(indices, offsets), tensor([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0], [0.0, 0.0]], device:), "padding excluded from mean denominator")
  padded(indices, offsets).sum.backward!
  close(padded.weight.grad, tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], device:), "padding gradients excluded")
  negative = initial * -1
  close(f.embedding_bag(indices, negative, offsets, mode: :max),
    tensor([[-1.0, -2.0], [0.0, 0.0], [-3.0, -4.0], [0.0, 0.0]], device:), "negative max and empty bags")

  samples = tensor([0.5, 2.0, 3.0, 1.0], device:, requires_grad: true)
  weighted = EmbeddingBag(4, 2, weight: initial, mode: :sum)
  close(weighted(indices, offsets, per_sample_weights: samples),
    tensor([[6.5, 9.0], [0.0, 0.0], [14.0, 18.0], [0.0, 0.0]], device:), "per-sample weighted sum")
  weighted(indices, offsets, per_sample_weights: samples).sum.backward!
  close(samples.grad, tensor([3.0, 7.0, 7.0, 11.0], device:), "per-sample weight derivative")
  close(weighted.weight.grad, tensor([[0.5, 0.5], [5.0, 5.0], [1.0, 1.0], [0.0, 0.0]], device:), "weighted parameter derivative")
  matrix = indices.reshape([2, 2])
  close(f.embedding_bag(matrix, initial, mode: :sum, include_last_offset: true),
    tensor([[4.0, 6.0], [8.0, 10.0]], device:), "2D bags ignore include-last-offset")
  close(f.embedding_bag(matrix, initial, mode: :sum, per_sample_weights: samples.reshape([2, 2])),
    tensor([[6.5, 9.0], [14.0, 18.0]], device:), "2D per-sample weights")
  close(f.embedding_bag(indices, initial, tensor([0, 2, 4], device:, dtype: "int64"), mode: :sum, include_last_offset: true),
    tensor([[4.0, 6.0], [8.0, 10.0]], device:), "CSR last offset")
  close(f.embedding_bag(indices.to(device:, dtype: "int32"), initial, offsets.to(device:, dtype: "int32")),
    mean_expected, "int32 indices and offsets")
  close(f.embedding_bag(zeros([0], device:, dtype: "int64"), initial, zeros([0], device:, dtype: "int64")),
    zeros([0, 2], device:), "zero bags")
  close(f.embedding_bag(zeros([0], device:, dtype: "int64"), initial, zeros([3], device:, dtype: "int64")),
    zeros([3, 2], device:), "all empty bags")
  strided_weight = tensor([[1.0, 3.0, 5.0, 9.0], [2.0, 4.0, 6.0, 10.0]], device:).transpose(0, 1)
  close(f.embedding_bag(indices.snapshot, strided_weight.snapshot, offsets.snapshot, mode: :sum),
    sum_expected, "noncontiguous snapshot weights")
  strided_indices = tensor([[0, 1, 2], [1, 2, 0]], device:, dtype: "int64").transpose(0, 1)
  close(f.embedding_bag(strided_indices, initial, mode: :sum),
    tensor([[4.0, 6.0], [8.0, 10.0], [6.0, 8.0]], device:), "noncontiguous 2D indices")

  lookup = Embedding.from_pretrained(initial, freeze: false, scale_grad_by_freq: true)
  lookup(indices).sum.backward!
  close(lookup.weight.grad, tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], device:), "embedding frequency-scaled gradients")
  frequency = EmbeddingBag(4, 2, weight: initial, mode: :sum, scale_grad_by_freq: true)
  frequency(tensor([0, 0, 1, 2], device:, dtype: "int64"), tensor([0], device:, dtype: "int64")).sum.backward!
  close(frequency.weight.grad, tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], device:), "bag frequency-scaled gradients")
  frequency_mean = EmbeddingBag(4, 2, weight: initial, scale_grad_by_freq: true, padding_idx: 1)
  close(frequency_mean(indices, offsets), padded(indices, offsets), "frequency scaling preserves padded forward")
  frequency_mean(indices, offsets).sum.backward!
  close(frequency_mean.weight.grad, padded.weight.grad, "frequency scaling with mean padding")
  frequency_samples = tensor([0.5, 2.0, 3.0, 1.0], device:, requires_grad: true)
  frequency_weighted = EmbeddingBag(4, 2, weight: initial, mode: :sum, scale_grad_by_freq: true)
  frequency_weighted(indices, offsets, per_sample_weights: frequency_samples).sum.backward!
  close(frequency_weighted.weight.grad, tensor([[0.5, 0.5], [2.5, 2.5], [1.0, 1.0], [0.0, 0.0]], device:), "weighted frequency gradient")
  close(frequency_samples.grad, samples.grad, "frequency scaling does not alter sample derivatives")
  frozen = Embedding.from_pretrained(initial.snapshot, padding_idx: -1)
  check(not frozen.weight.requires_grad? and frozen.weight.leaf? and not frozen.weight.readonly?, "pretrained owned frozen leaf")
  close(frozen(tensor([3], device:, dtype: "int64")), initial[3].unsqueeze(0), "pretrained padding value preserved")
  no_grad(): frozen.weight.fill!(0)
  close(initial, tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [9.0, 10.0]], device:), "pretrained never aliases input weight")
  random_padding = Embedding(4, 2, padding_idx: -1, device:)
  check(random_padding.padding_idx == 3, "negative padding index normalized")
  close(random_padding.weight[3], zeros([2], device:), "random padding row initialized to zero")
  random_padding.reset_parameters!(seed: 5)
  close(random_padding.weight[3], zeros([2], device:), "reset restores zero padding")
  restored = EmbeddingBag(4, 2, mode: :sum, device:)
  restored.load_state_dict!(sum_layer.state_dict)
  close(restored(indices, offsets), sum_expected, "bag state round trip")
  check(restored.parameters.count == 1 and restored.weight.leaf?, "bag parameter registry")

  norm_initial = tensor([[3.0, 4.0], [0.0, 6.0], [1.0, 0.0], [9.0, 12.0]], device:)
  norm_indices = tensor([0, 0, 1, 2], device:, dtype: "int64")
  norm_expected = tensor([[1.2, 1.6], [0.0, 2.0], [1.0, 0.0], [9.0, 12.0]], device:)
  bounded = Embedding(4, 2, weight: norm_initial, max_norm: 2)
  rejected(): bounded(norm_indices)
  bounded.call!(norm_indices).sum.backward!
  close(bounded.weight, norm_expected, "max-norm selected rows with duplicates")
  check(bounded.weight.leaf?, "renorm retains leaf parameter")
  close(bounded.weight.grad, tensor([[2.0, 2.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], device:), "renorm excluded from derivative")
  bounded_bag = EmbeddingBag(4, 2, weight: norm_initial, max_norm: 2, mode: :sum)
  rejected(): bounded_bag(norm_indices, tensor([0], device:, dtype: "int64"))
  bounded_bag.call!(norm_indices, tensor([0], device:, dtype: "int64"))
  close(bounded_bag.weight, norm_expected, "bag max norm")
  detached_norm = f.embedding_renormed(norm_initial.snapshot, norm_indices, 2)
  close(detached_norm, norm_expected, "non-mutating embedding_renormed")
  close(norm_initial, tensor([[3.0, 4.0], [0.0, 6.0], [1.0, 0.0], [9.0, 12.0]], device:), "renormed preserves source")
  l1_rows = norm_initial.copy
  f.embedding_renorm!(l1_rows, norm_indices, 2, norm_type: 1)
  close(l1_rows[0], tensor([6.0 / 7, 8.0 / 7], device:), "max-norm p1")
  rejected(): f.embedding_renorm!(norm_initial.snapshot, norm_indices, 2)
  rejected(): f.embedding(norm_indices, norm_initial, max_norm: 2)
  rejected(): f.embedding_bag(norm_indices, norm_initial, tensor([0], device:, dtype: "int64"), max_norm: 2)

  if device == "cpu":
    sparse_layer = Embedding(4, 2, weight: initial, sparse: true, padding_idx: 1)
    optimizer = SGD(sparse_layer.parameters, lr: 0.1)
    sparse_layer(indices).sum.backward!
    check(sparse_layer.weight.grad.is_sparse and sparse_layer.weight.grad.sparse?, "sparse embedding gradient")
    expected_sparse = tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], device:)
    close(sparse_layer.weight.grad.to_dense, expected_sparse, "sparse padding gradient values")
    sparse_bag = EmbeddingBag(4, 2, weight: initial, mode: :sum, sparse: true)
    sparse_bag(indices, offsets).sum.backward!
    check(sparse_bag.weight.grad.is_sparse, "sparse bag gradient")
    close(sparse_bag.weight.grad.densified, sum_layer.weight.grad, "densified bag gradient")
    check(not sparse_bag.weight.grad.snapshot.to_dense.readonly?, "sparse densification owns storage")
    rejected(): optimizer.step!
    sparse_layer.weight.grad!(sparse_layer.weight.grad.densified)
    optimizer.step!
    close(sparse_layer.weight, initial - expected_sparse * 0.1, "explicit dense gradient optimizer bridge")
    check(Embedding(4, 2, dtype: "float64").weight.dtype.to_str == "float64", "embedding constructor dtype")
  else:
    rejected(): Embedding(4, 2, weight: initial, sparse: true)(indices)
    rejected(): EmbeddingBag(4, 2, weight: initial, sparse: true)(indices, offsets)
  check(initial.snapshot.to_dense.readonly?, "dense conversion preserves read-only aliases")

  rejected(): Embedding(4, 2, padding_idx: -5, device:)
  rejected(): EmbeddingBag(4, 2, mode: :max, sparse: true, device:)
  rejected(): f.embedding(indices, initial, sparse: true, scale_grad_by_freq: true)
  rejected(): f.embedding_bag(indices, initial)
  rejected(): f.embedding_bag(matrix, initial, offsets)
  rejected(): f.embedding_bag(indices, initial, offsets, mode: :invalid)
  rejected(): f.embedding_bag(indices, initial, offsets, per_sample_weights: samples)
  rejected(): f.embedding_bag(matrix, initial, mode: :sum, per_sample_weights: samples)
  rejected(): f.embedding_bag(indices, initial, tensor([1, 4], device:, dtype: "int64"))
  rejected(): f.embedding_bag(indices, initial, tensor([0, 3, 2], device:, dtype: "int64"))
  rejected(): f.embedding_bag(indices, initial, tensor([0, 5], device:, dtype: "int64"))
  rejected(): f.embedding_bag(indices, initial, tensor([0, 2], device:, dtype: "int64"), include_last_offset: true)
  rejected(): f.embedding_bag(tensor([4], device:, dtype: "int64"), initial, tensor([0], device:, dtype: "int64"))
  rejected(): f.embedding(tensor([-1], device:, dtype: "int64"), initial)
  rejected(): f.embedding(indices.to(device:, dtype: "float32"), initial)
  rejected(): f.embedding_bag(indices, initial, offsets.to(device:, dtype: "int32"))
  unchanged = norm_initial.copy
  rejected(): f.embedding_bag!(norm_indices, unchanged, tensor([1], device:, dtype: "int64"), max_norm: 2)
  close(unchanged, norm_initial, "invalid bang bags do not mutate weights")
  rejected(): f.embedding!(tensor([4], device:, dtype: "int64"), unchanged, max_norm: 2)
  close(unchanged, norm_initial, "invalid bang lookup does not mutate weights")
  bounded.eval!
  rejected(): bounded(norm_indices)
  close(sum_layer(indices, offsets), sum_expected, "valid MPS call after invalid indices")
  detached = no_grad(): sum_layer(indices, offsets)
  inferred = inference_mode(): lookup(indices)
  check(not detached.requires_grad? and not inferred.requires_grad?, "embedding execution scopes")
  check(sum_layer.weight.grad.device.type == device and samples.grad.device.type == device and bounded.weight.device.type == device,
    "embedding backend placement")
  print "PASS torch embeddings " + device
  0

def main():
  parser = ArgParser(name: "Embedding checks")
  parser.arg("--device", choices: ["cpu", "mps"], default: "cpu")
  run(parser.parse_or_raise["device"])
