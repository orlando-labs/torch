package package_smoke

from torch.tensor.factories import tensor
from torch.backends import mps_available?
export main

def check(condition, message):
  raise ValueError(message) unless condition

def run(device):
  x = tensor([1.0, 2.0, 3.0], device:, requires_grad: true)
  check(x.device.type == device, "actual tensor device")
  loss = x.square.sum
  loss.backward!
  check(loss.item == 14.0, "forward")
  check(x.grad.to_a == [2.0, 4.0, 6.0], "backward")
  print("PASS package " + device + " forward=14 gradient=[2,4,6]")

def main():
  check(mps_available?, "MPS must be available; CPU fallback is disabled")
  run("cpu")
  run("mps")
  0
