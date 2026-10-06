package sputniktorch.idioms_test

import sputniktorch
import task
from sputniktorch.autograd import no_grad, with_grad, inference_mode, grad_enabled?
from sputniktorch.autograd import grad
from sputniktorch.autograd.context import execution_mode, with_trace
from sputniktorch.checks import check

export main

def main():
  x = sputniktorch.tensor([1.0, 2.0], requires_grad: true)
  check((x * 2 + 1).to_a == [3.0, 5.0], "Sputnik operators + implicit nullary calls")
  detached = no_grad(): (x ** 2).sum()
  check(not detached.requires_grad?, "scoped no_grad")
  check(grad_enabled?(), "scope restoration")
  with_trace(5):
    check(execution_mode() == 20 and grad_enabled?(), "trace token preserves grad mode")
    no_grad():
      check(execution_mode() == 21 and not grad_enabled?(), "no_grad preserves trace token")
      with_grad(): check(execution_mode() == 20, "nested with_grad preserves trace token")
      inference_mode(): check(execution_mode() == 22, "inference mode preserves trace token")
    check(execution_mode() == 20, "nested execution modes restore trace scope")
  check(execution_mode() == 0, "trace scope restores caller mode")
  nested = no_grad():
    with_grad(): x.square.sum()
  check(nested.requires_grad?, "nested with_grad")
  inferred = inference_mode(): x.square
  check(not inferred.requires_grad?, "inference mode")
  check(x.unsqueeze(0).shape == [1, 2], "shape operations")
  restored = false
  try:
    no_grad(): raise ValueError("scope unwind")
  rescue ValueError |error|:
    restored = grad_enabled?()
  check(restored, "exception-safe context restoration")
  detached_task = task.spawn:
    no_grad():
      task.sleep(0.01)
      local = sputniktorch.ones([8], requires_grad: true)
      not (local * 2).requires_grad?
  tracked_task = task.spawn:
    with_grad():
      task.sleep(0.01)
      local = sputniktorch.ones([8], requires_grad: true)
      (local * 2).requires_grad?
  check(detached_task.wait and tracked_task.wait, "grad context follows logical tasks")
  check(grad_enabled?(), "child tasks do not rebind parent")
  first = grad([x.square.sum()], [x], create_graph: true, retain_graph: true)[0]
  check(first.to_a == [2.0, 4.0] and x.grad == null, "functional gradients do not accumulate")
  second = grad([first.sum()], [x])[0]
  check(second.to_a == [2.0, 2.0], "higher-order differentiation")
  seed_gradient = sputniktorch.ones_like(x).snapshot
  x.grad!(seed_gradient)
  x.grad.mul!(2)
  check(seed_gradient.to_a == [1.0, 1.0], "grad! copies values without granting writable aliases")
  x.grad!(null)
  check(x.grad == null, "explicit gradient reset")
  print "PASS Sputnik idioms + task-local autograd"
  0
