package sputniktorch.mnist_hotspot

import sputniktorch
from sputniktorch.backends import execution_batch
from sputniktorch.data import DataLoader
from sputniktorch.random.generator import Generator
from sputniktorch.mnist.dataset import MNIST
from sputniktorch.mnist.network import SmallResNet
from sputniktorch.mnist.batch_profile import elapsed, synchronize

export main

def main():
  parser = ArgParser(name: "Isolated MNIST hotspot")
  parser.arg("--case", choices: ["batch", "epoch", "array", "forward", "forward_mutating", "stateful"], default: "batch")
  parser.arg("--iterations", type: Int, default: 500000)
  parser.arg("--data", default: "build/datasets/mnist")
  parser.arg("--device", choices: ["cpu", "mps"], default: "cpu")
  options = parser.parse_or_raise
  iterations = options["iterations"]
  raise ValueError("iterations must be positive") if iterations <= 0
  device = options["device"]
  result = execution_batch():
    if options["case"] == "array":
      order = Generator(seed: 7).randperm!(2048)
      value = null
      10.times: order.to_a
      started = Time.monotonic
      iterations.times:
        value = order.to_a
      {seconds: elapsed(started), checksum: value.sum}
    else if options["case"] == "batch":
      dataset = MNIST(options["data"], split: :train, limit: 2048)
      indices = Array.of(32): $it
      value = null
      100.times: dataset.batch(indices)
      started = Time.monotonic
      iterations.times:
        value = dataset.batch(indices)
      {seconds: elapsed(started), checksum: value[1].numel}
    else if options["case"] == "epoch":
      dataset = MNIST(options["data"], split: :train, limit: 2048)
      batches = DataLoader(dataset, batch_size: 32, shuffle: true, seed: 7)
      10.times:
        batches.each |batch|: batch[1].numel
      synchronize("cpu")
      count = 0
      started = Time.monotonic
      iterations.times:
        count = 0
        batches.each |batch|: count += batch[1].numel
      {seconds: elapsed(started), checksum: count}
    else:
      model = SmallResNet(device:)
      model.eval!
      x = sputniktorch.ones([32, 1, 28, 28], device:)
      sputniktorch.no_grad:
        value = null
        if options["case"] == "stateful":
          10.times: model.stateful?
          started = Time.monotonic
          iterations.times:
            value = model.stateful?
          {seconds: elapsed(started), checksum: if value then 1 else 0}
        else:
          if options["case"] == "forward_mutating":
            10.times: model.call!(x)
            synchronize(device)
            started = Time.monotonic
            iterations.times:
              value = model.call!(x)
            synchronize(device)
            {seconds: elapsed(started), checksum: value.sum.item}
          else:
            10.times: model(x)
            synchronize(device)
            started = Time.monotonic
            iterations.times:
              value = model(x)
            synchronize(device)
            {seconds: elapsed(started), checksum: value.sum.item}
  print "MNIST_HOTSPOT #{Json.generate(result)}"
  0
