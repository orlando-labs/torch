package sputniktorch.modular_imports_test

from sputniktorch.nn import Linear, Sequential, ReLU
from sputniktorch.optim import SGD
from sputniktorch.tensor.factories import tensor
from sputniktorch.checks import check

export main

def main():
  model = Sequential([Linear(2, 3, seed: 7), ReLU(), Linear(3, 1, seed: 8)])
  x = tensor([[1.0, 2.0], [3.0, 4.0]])
  check(model.call(x).shape == [2, 1], "modular imports + Sequential")
  optimizer = SGD(model.parameters)
  optimizer.zero_grad!
  model.call(x).square.mean().backward!()
  optimizer.step!
  check(model.state_dict.count == 4, "nested parameter names")
  print("PASS modular imports + nested model")
  0
