package sputniktorch.mnist_benchmark

import sputniktorch
from sputniktorch.backends import num_threads, num_interop_threads, execution_batch
from sputniktorch.nn import CrossEntropyLoss
from sputniktorch.optim import AdamW
from sputniktorch.optim.lr_scheduler import CosineAnnealingLR
from sputniktorch.data import DataLoader
from sputniktorch.mnist.dataset import MNIST
from sputniktorch.mnist.network import SmallResNet
from sputniktorch.mnist.training import train_epoch, evaluate
from sputniktorch.mnist.benchmark_options import BenchmarkOptions
from sputniktorch.mnist.batch_profile import BatchProfile, elapsed, synchronize

export main

def main():
  options = BenchmarkOptions()
  if options.execution_scope == "run":
    execution_batch:
      run(options)
  else:
    run(options)

def run(options):
  started = Time.monotonic
  device = options.device
  phase = Time.monotonic
  train = MNIST(options.data, split: :train, limit: options.train_limit)
  test = MNIST(options.data, split: :test, limit: options.test_limit)
  data_seconds = elapsed(phase)
  phase = Time.monotonic
  batches = DataLoader(train, batch_size: options.batch_size, shuffle: true, seed: options.seed)
  test_batches = DataLoader(test, batch_size: options.batch_size)
  model = SmallResNet(seed: options.seed, device:)
  loss_fn = CrossEntropyLoss()
  optimizer = AdamW(model.parameters, lr: 0.003, weight_decay: 0.0001)
  schedule = CosineAnnealingLR(optimizer, options.epochs, eta_min: 0.0003)
  parameters = model.parameters.reduce(0): _1 + _2.numel
  synchronize(device)
  setup_seconds = elapsed(phase)
  epochs = []

  options.epochs.times |epoch|:
    batches.sampler.set_epoch!(epoch)
    profile = if options.profile then BatchProfile(device) else null
    synchronize(device)
    phase = Time.monotonic
    train_loss = train_epoch(model, batches, loss_fn, optimizer, device:, profile:)
    synchronize(device)
    train_seconds = elapsed(phase)
    phase = Time.monotonic
    (test_loss, accuracy) = evaluate(model, test_batches, loss_fn, device:)
    synchronize(device)
    eval_seconds = elapsed(phase)
    epochs.push!({epoch: epoch + 1, train_loss:, test_loss:, accuracy:,
      train_seconds:, eval_seconds:,
      profile: (if profile == null then null else profile.seconds)})
    print "epoch #{epoch + 1}/#{options.epochs} | train #{train_loss} | test #{test_loss} | accuracy #{accuracy * 100}%"
    schedule.step!

  phase = Time.monotonic
  sputniktorch.save(model.state_dict, options.checkpoint)
  synchronize(device)
  save_seconds = elapsed(phase)
  phase = Time.monotonic
  restored = SmallResNet(seed: options.seed, device:)
  restored.load_state_dict!(sputniktorch.load(options.checkpoint, map_location: device))
  synchronize(device)
  restore_seconds = elapsed(phase)
  phase = Time.monotonic
  (loss, accuracy) = evaluate(restored, test_batches, loss_fn, device:)
  synchronize(device)
  restore_eval_seconds = elapsed(phase)
  final = epochs[-1]
  changed = (loss - final[:test_loss]).abs > 1e-7 or accuracy != final[:accuracy]
  raise ValueError("checkpoint changed test metrics") if changed
  result = {schema: "sputniktorch.mnist.sample.v1", device:, version: sputniktorch.version(),
    threads: num_threads(), interop_threads: num_interop_threads(),
    train_count: train.count, test_count: test.count, parameters:, batch_size: options.batch_size,
    seed: options.seed, optimizer: "matched", profiled: options.profile,
    execution_scope: options.execution_scope,
    data_seconds:, setup_seconds:, epochs:, save_seconds:, restore_seconds:,
    restore_eval_seconds:, checkpoint_loss: loss, checkpoint_accuracy: accuracy,
    workload_seconds: elapsed(started)}
  print "MNIST_BENCH #{Json.generate(result)}"
  0
