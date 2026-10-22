package torch.materialization_test

import torch
from torch.checks import check

export main

def run(device):
  names = ["bool", "uint8", "int8", "int16", "int32", "int64",
    "float16", "bfloat16", "float32"]
  names.push!("float64") if device == "cpu"
  names.each |dtype|:
    data = if dtype == "bool" then [[true, false, true], [false, true, false]] else [[1, 2, 3], [4, 5, 6]]
    x = torch.tensor(data, dtype:, device:)
    check(x.to_a == data, "#{dtype} contiguous materialization")
    expected = if dtype == "bool" then [[false, true], [true, false]] else [[2, 5], [3, 6]]
    view = x.transpose(0, 1).narrow(0, 1, 2)
    check(view.to_a == expected, "#{dtype} strided materialization with storage offset")
    scalar = if dtype == "bool" then true else 3
    check(torch.tensor(scalar, dtype:, device:).to_a == scalar, "#{dtype} scalar materialization")
    check(torch.zeros([2, 0, 3], dtype:, device:).to_a == [[], []], "#{dtype} nested empty dimensions")
    check(torch.zeros([0, 2], dtype:, device:).to_a == [], "#{dtype} empty leading dimension")
    value = x.to_a[0][0]
    if dtype == "bool":
      check(Bool === value, "bool remains Bool")
    else if dtype in ["float16", "bfloat16", "float32", "float64"]:
      check(Float === value, "#{dtype} remains Float")
    else:
      check(Int === value, "#{dtype} remains Int")
  large = [9007199254740993, -9007199254740993, 9223372036854775807]
  check(torch.tensor(large, dtype: "int64", device:).to_a == large,
    "int64 materialization preserves values beyond exact double precision")
  check(torch.tensor([0.5, -1.25], dtype: "float64").to_a == [0.5, -1.25],
    "fractional double materialization")
  print "PASS tensor materialization on #{device}"
  0

def main():
  parser = ArgParser(name: "Tensor materialization checks")
  parser.arg("--device", choices: ["cpu", "mps"], default: "cpu")
  run(parser.parse_or_raise["device"])
