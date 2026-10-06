package sputniktorch.autograd.functional

from sputniktorch.tensor.value import Tensor
from sputniktorch.autograd.context import execution_mode

export grad

native def _grad(outputs, inputs, gradients, retain_graph as Bool, create_graph as Bool,
  allow_unused as Bool, mode as Int) from "amt_grad"

# Functional gradients do not accumulate into input.grad. Lists keep the
# correspondence between inputs and results explicit, including unused nulls.

def grad(outputs, inputs, gradients: null, retain_graph: false, create_graph: false, allow_unused: false):
  output_handles = outputs.map: $it._core
  input_handles = inputs.map: $it._core
  gradient_handles = if gradients == null:
    null
  else:
    gradients.map: if $it == null then null else $it._core
  result = _grad(output_handles, input_handles, gradient_handles,
    retain_graph, create_graph, allow_unused, execution_mode())
  result.map:
    if null === $it then null else Tensor($it)
