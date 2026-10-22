package torch.mps_test

from torch.checks import run
from torch.checks import check
import torch
export main
def main():
  run("mps")
  state = torch.load("build/fixtures/cuda_views.pt", map_location: "mps")
  check(state["left"].device.type == "mps", "checkpoint actual MPS placement")
  check(state["left"].to_a == [[1.0, 2.0], [4.0, 5.0]], "CUDA to MPS checkpoint")
  state["left"].fill!(9)
  check(state["right"].to_a == [[9.0, 3.0], [9.0, 6.0]], "MPS shared checkpoint storage")
  print("PASS map_location CUDA to MPS")
  0
