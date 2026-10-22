package torch.serialization_test

import torch
from torch.checks import check

export main

def main():
  state = {"weight": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
  torch.save(state, "build/fixtures/sputnik.pt")
  restored = torch.load("build/fixtures/sputnik.pt")
  check(restored["weight"].to_a == state["weight"].to_a, "save/load round trip")
  views = torch.load("build/fixtures/cuda_views.pt")
  check(views["left"].device.type == "cpu", "default CPU placement of CUDA-tagged storage")
  check(views["left"].to_a == [[1.0, 2.0], [4.0, 5.0]], "strided left view")
  check(views["right"].to_a == [[2.0, 3.0], [5.0, 6.0]], "offset right view")
  views["left"].fill!(9)
  check(views["right"].to_a == [[9.0, 3.0], [9.0, 6.0]], "shared storage survives load")
  mapped = torch.load("build/fixtures/cuda_views.pt", map_location: {"cuda:3": "cpu"})
  check(mapped["right"].to_a == [[2.0, 3.0], [5.0, 6.0]], "map_location dictionary")
  ordered = torch.load("build/fixtures/ordered_metadata.pt")
  check(ordered["left"].to_a == [[1.0, 2.0], [4.0, 5.0]], "OrderedDict metadata")
  framed = torch.load("build/fixtures/protocol4.pt")
  check(framed["right"].to_a == mapped["right"].to_a, "protocol 4 FRAME")
  ["bad_bounds", "forbidden_global", "truncated", "unsupported", "duplicate"].each |name|:
    rejected = false
    try:
      torch.load("build/fixtures/" + name + ".pt")
    rescue Torch.Errors.CheckpointError |error|:
      rejected = true
    check(rejected, "invalid checkpoint must be rejected")
  print("PASS torch state dict + map_location + restricted parser")
  0
