package torch.idioms_test

import torch
import task
from torch.autograd import no_grad, with_grad, inference_mode, grad_enabled?
from torch.autograd import grad
from torch.autograd.context import execution_mode, with_trace
from torch.checks import check
from torch.backends import execution_batch

export main

native def _legacy_execution_batch(block) from "amt_execution_batch" blocking

def main():
  x = torch.tensor([1.0, 2.0], requires_grad: true)
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
  no_grad():
    execution_batch():
      check(not x.relu.requires_grad?, "batch installs inherited no_grad")
      with_grad():
        check(x.relu.requires_grad?, "nested grad overrides batch carrier state")
      check(not x.relu.requires_grad?, "nested grad restores batch no_grad")
      nested_batch = inference_mode():
        execution_batch(): x.relu
      check(not nested_batch.requires_grad?, "nested inference batch")
      failed = false
      try:
        with_grad(): x.reshape([3])
      rescue Torch.Errors.Error:
        failed = true
      check(failed and not x.relu.requires_grad?, "kernel fault restores batch no_grad")
  check(x.relu.requires_grad?, "batch restores caller carrier state")
  legacy = _legacy_execution_batch(| |: x.relu)
  check(legacy.requires_grad?, "one-argument batch thunk remains compatible")
  with_trace(5):
    execution_batch():
      check(execution_mode() == 20, "batch leaves tracing attachment to individual kernels")
  detached_task = task.spawn:
    no_grad():
      task.sleep(0.01)
      local = torch.ones([8], requires_grad: true)
      not (local * 2).requires_grad?
  tracked_task = task.spawn:
    with_grad():
      task.sleep(0.01)
      local = torch.ones([8], requires_grad: true)
      (local * 2).requires_grad?
  check(detached_task.wait and tracked_task.wait, "grad context follows logical tasks")
  batch_task = task.spawn:
    no_grad():
      failed = false
      try:
        execution_batch():
          check(not x.relu.requires_grad?, "task batch inherits no_grad")
          raise ValueError("task batch boundary")
      rescue ValueError:
        failed = true
      failed and not x.relu.requires_grad? and not grad_enabled?()
  check(batch_task.wait, "parked execution batch restores fault and task-local scope")
  check(grad_enabled?(), "child tasks do not rebind parent")
  first = grad([x.square.sum()], [x], create_graph: true, retain_graph: true)[0]
  check(first.to_a == [2.0, 4.0] and x.grad == null, "functional gradients do not accumulate")
  second = grad([first.sum()], [x])[0]
  check(second.to_a == [2.0, 2.0], "higher-order differentiation")
  seed_gradient = torch.ones_like(x).snapshot
  x.grad!(seed_gradient)
  x.grad.mul!(2)
  check(seed_gradient.to_a == [1.0, 1.0], "grad! copies values without granting writable aliases")
  x.grad!(null)
  check(x.grad == null, "explicit gradient reset")
  print "PASS Sputnik idioms + task-local autograd"
  0
