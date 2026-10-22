package torch.direct_tensor_import_test

from torch.tensor.factories import tensor

export main

def main():
  caught = false
  try:
    tensor([[1], [2, 3]])
  rescue Torch.Errors.ShapeError |error|:
    caught = true
  if not caught:
    raise ValueError("direct tensor import must register native errors")
  print("PASS direct tensor import + native errors")
  0
