package torch.training_test

import torch
import task
from torch.autograd import grad_enabled?
from torch.backends import execution_batch
from torch.optim import SGD, Adam, AdamW, clip_grad_norm!, clip_grad_value!
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.data import TensorDataset, Subset, ConcatDataset, DataLoader, RandomSampler, DistributedSampler, default_collate
from torch.data import Dataset, IterableDataset
from torch.checks import check

export main, run

def near(actual, expected): check((actual - expected).abs < 1e-5, "numeric reference: #{actual} != #{expected}")

class CountingStream < IterableDataset:
  def each(&block):
    5.times |index|: block(index)
    self

def run(device):
  abstract_refused = false
  try:
    Dataset().count
  rescue NoMethodError |error|:
    abstract_refused = true
  check(abstract_refused, "dataset base reports an unimplemented Sputnik method")
  p = torch.tensor([1.0], device:, requires_grad: true)
  optimizer = SGD([p, p], lr: 0.1, momentum: 0.9)
  check(optimizer.parameters.count == 1, "tied parameter deduplication")
  (p * 2).sum().backward!()
  optimizer.step!
  near(p.item, 0.8)
  optimizer.zero_grad!
  (p * 3).sum().backward!()
  optimizer.step!
  near(p.item, 0.32)
  check(p.leaf?, "optimizer preserves leaf identity")

  (p * 2).sum().backward!()
  gradient = p.grad
  optimizer.zero_grad_buffers!
  check(gradient.item == 0.0 and p.grad.item == 0.0, "zero existing gradient buffers")
  optimizer.zero_grad!
  check(p.grad == null, "bare zero_grad! releases buffers")

  p = torch.tensor([1.0], device:, requires_grad: true)
  betas = [0.5, 0.5]
  optimizer = Adam([p], lr: 0.1, betas:, eps: 1e-8)
  betas[0] = 0.9
  betas[1] = 0.9
  (p * 2).sum().backward!()
  optimizer.step!
  near(p.item, 0.9)
  optimizer.zero_grad!
  (p * 4).sum().backward!()
  optimizer.step!
  near(p.item, 0.8037749551)
  saved_optimizer = optimizer.state_dict
  saved_parameter = p.snapshot
  optimizer.zero_grad!
  (p * 5).sum().backward!()
  optimizer.step!
  continued = p.item
  restored_parameter = saved_parameter.copy.requires_grad!(true)
  restored_optimizer = Adam([restored_parameter], lr: 1.0, betas: (0.1, 0.1))
  restored_optimizer.load_state_dict!(saved_optimizer)
  (restored_parameter * 5).sum().backward!()
  restored_optimizer.step!
  near(restored_parameter.item, continued)
  check(restored_optimizer.parameters[0]._identity == restored_parameter._identity, "optimizer restore preserves parameter references")
  optimizer.zero_grad!
  optimizer.step!
  near(p.item, continued)

  p = torch.tensor([1.0], device:, requires_grad: true)
  optimizer = AdamW([p], lr: 0.1, weight_decay: 0.1)
  p.sum().backward!()
  optimizer.step!
  near(p.item, 0.89)

  # Packed updates must mutate the existing parameters and their views.
  packed_left = torch.tensor([1.0, -2.0], device:, requires_grad: true)
  packed_right = torch.tensor([3.0], device:, requires_grad: true)
  alias = packed_left.detach.reshape([1, 2])
  identity = packed_left._identity
  retained = packed_left.square.sum
  packed_optimizer = AdamW([packed_left, packed_right], lr: 0.1, weight_decay: 0.1)
  (packed_left.sum + packed_right.sum).backward!
  packed_optimizer.step!
  near(alias[0][0].item, 0.89)
  near(alias[0][1].item, -2.08)
  near(packed_right.item, 2.87)
  check(packed_left._identity == identity and packed_left.leaf?, "packed update preserves parameter identity")
  version_checked = false
  try:
    retained.backward!
  rescue Torch.Errors.Error |error|:
    version_checked = true
  check(version_checked, "packed update preserves autograd mutation checks")
  saved_packed = packed_optimizer.state_dict
  packed_optimizer.zero_grad!
  packed_left.sum.backward!
  packed_optimizer.step!
  check(packed_optimizer.state_dict[:steps] == [2, 1], "packed moments support a later missing gradient")
  check(saved_packed[:steps] == [1, 1], "packed checkpoint counters remain independent")
  near(saved_packed[:first][0][0].item, 0.1)

  storage = torch.tensor([1.0], device:)
  aliased_left = storage.detach.requires_grad!(true)
  aliased_right = storage.detach.requires_grad!(true)
  aliased = AdamW([aliased_left, aliased_right], lr: 0.1, weight_decay: 0.1)
  (aliased_left.sum + aliased_right.sum).backward!
  aliased.step!
  near(storage.item, 0.89 * 0.99 - 0.1)
  check(aliased.state_dict[:steps] == [1, 1], "aliased parameters preserve sequential AdamW updates")

  # A foreach update must preserve independent counters, absent gradients,
  # mixed dtypes and immutable optimizer snapshots.
  left = torch.tensor([1.0], device:, requires_grad: true)
  right_dtype = if device == "cpu" then "float64" else "float32"
  right = torch.tensor([3.0], device:, dtype: right_dtype, requires_grad: true)
  grouped = AdamW([left, right], lr: 0.1, betas: (0.5, 0.5), weight_decay: 0.1)
  left.grad!(torch.tensor([2.0], device:))
  grouped.step!
  near(left.item, 0.89)
  near(right.item, 3.0)
  first_state = grouped.state_dict
  check(first_state[:steps] == [1, 0], "missing gradients do not advance steps or decay parameters")
  check(first_state[:first][0].readonly?, "optimizer state is a read-only snapshot")
  grouped.zero_grad!
  right.grad!(torch.tensor([3.0], device:, dtype: right_dtype))
  grouped.step!
  near(left.item, 0.89)
  near(right.item, 2.87)
  check(grouped.state_dict[:steps] == [1, 1], "per-parameter counters remain independent")
  near(first_state[:first][1].item, 0.0)
  gradient_alias = right.grad
  grouped.zero_grad_buffers!
  near(gradient_alias.item, 0.0)
  check(right.grad != null, "grouped buffer reset retains gradient identity")
  grouped.zero_grad!
  left.grad!(torch.tensor([4.0], device:))
  right.grad!(torch.tensor([4.0], device:, dtype: right_dtype))
  grouped.step!
  near(left.item, 0.89 * 0.99 - 0.1 * (2.5 / 0.75) / ((9.0 / 0.75) ** 0.5 + 1e-8))
  near(right.item, 2.87 * 0.99 - 0.1 * (2.75 / 0.75) / ((10.25 / 0.75) ** 0.5 + 1e-8))
  grouped.zero_grad!
  before = grouped.state_dict
  grouped.step!
  check(grouped.state_dict[:steps] == before[:steps], "empty active group is a no-op")

  # The native callback keeps the logical task context and restores grad mode
  # when an Sputnik exception escapes the optimizer execution scope.
  context = task.local(inherit: true)
  context.with(17):
    result = execution_batch():
      check(context.get == 17 and grad_enabled?(), "execution batch preserves caller grad mode and task locals")
      grouped._run_step!():
        check(context.get == 17 and not grad_enabled?(), "optimizer worker inherits task-local context")
        grouped._run_step!():
          check(context.get == 17 and not grad_enabled?(), "nested optimizer scope does not deadlock")
      42
    check(result == 42, "execution batch returns the Sputnik block result")
  check(grad_enabled?(), "optimizer worker restores caller grad mode")
  caught = false
  try:
    grouped._run_step!(): raise ValueError("optimizer scope unwind")
  rescue ValueError |error|:
    caught = grad_enabled?()
  check(caught, "optimizer callback exception is rescuable and restores grad mode")
  grouped._run_step!(): check(not grad_enabled?(), "optimizer scope remains reusable after a failure")

  optimizer.zero_grad!
  (p * 4).sum().backward!()
  near(clip_grad_norm!([p, p], max_norm: 2.0), 4.0)
  near(p.grad.item, 2.0)
  clip_grad_value!([p], clip_value: 0.25)
  near(p.grad.item, 0.25)

  step = StepLR(optimizer, 2, gamma: 0.5)
  step.step!
  near(optimizer.lr, 0.1)
  step.step!
  near(optimizer.lr, 0.05)
  saved = step.state_dict
  step.step!.step!
  step.load_state_dict!(saved)
  near(optimizer.lr, 0.05)
  cosine = CosineAnnealingLR(optimizer, 2)
  cosine.step!
  near(optimizer.lr, 0.025)
  plateau = ReduceLROnPlateau(optimizer, patience: 0, factor: 0.5)
  plateau.step!(1.0).step!(1.1)
  near(optimizer.lr, 0.0125)

  x = torch.arange(5, device:)
  y = x * 2
  dataset = TensorDataset(x, y)
  check(dataset.count == 5, "tensor dataset length")
  loader = DataLoader(dataset, batch_size: 2)
  check(loader.count == 3, "partial batch")
  batches = loader.map |batch|: batch
  check(batches[0][0].to_a == [0, 1], "batch collate")
  check(batches[2][1].to_a == [8], "last batch")
  threaded = DataLoader(dataset, batch_size: 2, num_workers: 2, prefetch_factor: 1)
  threaded_batches = threaded.map |batch|: (batch[0].to_a, batch[1].to_a)
  check(threaded_batches == [([0, 1], [0, 2]), ([2, 3], [4, 6]), ([4], [8])],
    "threaded vectorized tensor batches preserve order and tail")
  check(DataLoader(dataset, batch_size: 2, drop_last: true).count == 2, "drop last")
  count_samples = |samples|: samples.count
  custom = DataLoader(dataset, batch_size: 2, collate: &count_samples)
  check((custom.map: $it) == [2, 2, 1], "custom collation bypasses tensor batching")
  subset = Subset(dataset, [4, 1])
  joined = ConcatDataset([subset, dataset])
  check(joined.count == 7 and joined[2][0].item == 0, "dataset composition")
  sampler = RandomSampler(20, seed: 8)
  first = sampler.indices
  check(first == sampler.indices, "sampler has no hidden mutation")
  sampler.set_epoch!(1)
  check(first != sampler.indices, "explicit epoch changes shuffle")
  left = DistributedSampler(5, rank: 0, world_size: 2, shuffle: false)
  right = DistributedSampler(5, rank: 1, world_size: 2, shuffle: false)
  check(left.indices == [0, 2, 4] and right.indices == [1, 3, 0], "distributed padding")
  check(DistributedSampler(5, rank: 0, world_size: 2, shuffle: false, drop_last: true).indices == [0, 2], "distributed truncation")
  collated = default_collate([{x: 1, y: 2}, {x: 3, y: 4}])
  check(collated[:x].to_a == [1, 3], "nested map collate")
  stream = DataLoader(CountingStream(), batch_size: 2)
  streamed = stream.map: $it.to_a
  check(streamed == [[0, 1], [2, 3], [4]], "streaming collation")
  print "PASS training and data on #{device}"
  0

def main(): run("cpu")
