package torch.mnist_atoms

import torch
from torch.autograd.context import execution_mode
from torch.backends import execution_batch
from torch.tensor.core import unary
from torch.tensor.value import Tensor
from torch.data import DataLoader
from torch.nn import Conv2d, CrossEntropyLoss
from torch.optim import AdamW
from torch.nn.functional import kernel_cores
from torch.random.generator import Generator
from torch.mnist.dataset import MNIST
from torch.mnist.network import SmallResNet
from torch.mnist.batch_profile import elapsed, synchronize

export main

class A:
  def call(x): x
class B < A:
  def call(x): x
class C < A:
  def call(x): x
class D < A:
  def call(x): x
class E < A:
  def call(x): x
class F < A:
  def call(x): x

# Every repetition includes the same closure/loop overhead. Keep the final
# result alive until after synchronization, and check it outside the timer.
def measure(name, iterations, device, &block):
  rows = []
  value = null
  6.times |repeat|:
    execution_batch:
      synchronize(device)
      started = Time.monotonic
      iterations.times:
        value = block()
      synchronize(device)
      seconds = elapsed(started)
      rows.push!(seconds) if repeat > 0
  checksum = if Tensor === value then value.sum.item else value
  print "MNIST_ATOM #{Json.generate({name:, iterations:, seconds: rows, checksum:})}"

def main():
  parser = ArgParser(name: "MNIST atomic costs")
  parser.arg("--device", choices: ["cpu", "mps"], default: "cpu")
  parser.arg("--data", default: "build/datasets/mnist")
  args = parser.parse_or_raise
  device = args["device"]
  x = torch.ones([32, 1, 28, 28], device:)
  core = x._core
  model = SmallResNet(device:)
  model.eval!
  conv = Conv2d(1, 16, 3, padding: 1, bias: false, seed: 7, device:)
  weight = conv.weight._core
  options = [[1, 1], [1, 1], [1, 1], 1]
  mono = [A(), A(), A(), A(), A(), A()]
  poly = [A(), B(), C(), D(), E(), F()]
  measure("loop", 100000, "cpu"): 42
  measure("mono_dispatch", 10000, "cpu"):
    mono.reduce(42) |value, layer|: layer(value)
  measure("poly_dispatch", 10000, "cpu"):
    poly.reduce(42) |value, layer|: layer(value)
  measure("execution_mode", 100000, "cpu"): execution_mode()
  measure("core_numel", 100000, "cpu"): core.numel
  measure("tensor_numel", 100000, "cpu"): x.numel
  order = Generator(seed: 7).randperm!(2048)
  measure("sampler_to_array", 1000, "cpu"): order.to_a[0]
  training_model = SmallResNet(device:)
  labels = torch.zeros([32], dtype: "int64", device:)
  loss_fn = CrossEntropyLoss()
  optimizer = AdamW(training_model.parameters, lr: 0.003, weight_decay: 0.0001)
  measure("train_step_preloaded", 100, device):
    optimizer.zero_grad!
    loss = loss_fn(training_model.call!(x), labels)
    loss.backward!
    optimizer.step!
    loss.item
  parameters = model.parameters
  parameters.each |parameter|: parameter.grad!(torch.ones_like(parameter))
  isolated_optimizer = AdamW(parameters, lr: 0.003, weight_decay: 0.0001)
  measure("optimizer_step", 500, device):
    isolated_optimizer.step!
    42
  # The optimizer above changes model's parameters; use a fresh matching model
  # for forward comparisons.
  model = SmallResNet(device:)
  model.eval!
  torch.no_grad:
    measure("relu_core", 10000, device): Tensor(unary(core, "relu", null))
    measure("relu_tensor", 10000, device): x.relu
    measure("conv_core", 300, device): kernel_cores("conv2d", [core, weight, null], options)
    measure("conv_module", 300, device): conv(x)
    measure("model_eval", 100, device): model(x)
  dataset = MNIST(args["data"], split: :train, limit: 2048)
  indices = Array.of(32): $it
  batches = DataLoader(dataset, batch_size: 32, shuffle: true, seed: 7)
  measure("dataset_batch", 1000, "cpu"):
    dataset.batch(indices)[1].numel
  measure("data_epoch", 20, "cpu"):
    seen = 0
    batches.each |(features, labels)|: seen += labels.numel
    seen
  0
