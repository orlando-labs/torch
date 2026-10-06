package sputniktorch.data.collate

from sputniktorch.tensor.value import Tensor
from sputniktorch.tensor.factories import tensor
from sputniktorch.nn.functional import stack

export default_collate

def default_collate(samples):
  raise ValueError("cannot collate an empty batch") if samples.empty?
  first = samples[0]
  if Tensor === first:
    stack(samples)
  else if Array === first or Tuple === first:
    values = Array.of(first.count) |index|:
      column = samples.map: $it[index]
      default_collate(column)
    if Tuple === first then values.tuple else values
  else if Map === first:
    first.transform |name, value|:
      column = samples.map: $it[name]
      (name, default_collate(column))
  else:
    tensor(samples)
