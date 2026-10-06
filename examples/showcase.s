package sputniktorch.showcase

import sputniktorch
from sputniktorch.nn import CrossEntropyLoss
from sputniktorch.optim import AdamW
from sputniktorch.optim.lr_scheduler import CosineAnnealingLR
from sputniktorch.data import DataLoader
from sputniktorch.mnist.dataset import MNIST
from sputniktorch.mnist.network import SmallResNet
from sputniktorch.mnist.options import Options
from sputniktorch.mnist.training import train_epoch, evaluate

export main

def main():
  options = Options()
  device = options.device
  train = MNIST(options.data, split: :train, limit: options.train_limit)
  test = MNIST(options.data, split: :test, limit: options.test_limit)
  batches = DataLoader(train, batch_size: options.batch_size, shuffle: true, seed: options.seed)
  test_batches = DataLoader(test, batch_size: options.batch_size)
  model = SmallResNet(seed: options.seed, device:)
  loss_fn = CrossEntropyLoss()
  optimizer = AdamW(model.parameters, lr: 0.003, weight_decay: 0.0001)
  schedule = CosineAnnealingLR(optimizer, options.epochs, eta_min: 0.0003)
  parameters = model.parameters.reduce(0): _1 + _2.numel
  print "MNIST | #{device} | train #{train.count} | test #{test.count} | parameters #{parameters}"
  final_metrics = null

  options.epochs.times |epoch|:
    started = Time.monotonic
    batches.sampler.set_epoch!(epoch)
    train_loss = train_epoch(model, batches, loss_fn, optimizer, device:)
    (test_loss, accuracy) = evaluate(model, test_batches, loss_fn, device:)
    final_metrics = (test_loss, accuracy)
    seconds = (Time.monotonic - started).total_nanoseconds.float / 1e9
    print("epoch #{epoch + 1}/#{options.epochs}" +
      " | train loss #{train_loss} | test loss #{test_loss}" +
      " | accuracy #{accuracy * 100}% | #{seconds}s")
    schedule.step!

  sputniktorch.save(model.state_dict, options.checkpoint)
  restored = SmallResNet(seed: options.seed, device:)
  restored.load_state_dict!(sputniktorch.load(options.checkpoint, map_location: device))
  (loss, accuracy) = evaluate(restored, test_batches, loss_fn, device:)
  changed = (loss - final_metrics[0]).abs > 1e-7 or accuracy != final_metrics[1]
  raise ValueError("checkpoint changed test metrics") if changed
  print "checkpoint #{options.checkpoint} | test loss #{loss} | accuracy #{accuracy * 100}%"
  0
