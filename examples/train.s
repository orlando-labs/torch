package torch.train_example

import torch
export main

def main():
  model = torch.Linear(1, 1, seed: 7, device: "mps")
  optimizer = torch.SGD(model.parameters, lr: 0.05)
  x = torch.tensor([[1.0], [2.0], [3.0]], device: "mps")
  target = x.mul(2).add(1)
  200.times:
    optimizer.zero_grad!
    loss = model.call(x).sub(target).square.mean()
    loss.backward!()
    optimizer.step!
  print(model.call(x).to_a)
  print(model.call(x).sub(target).square.mean().item)
  0
