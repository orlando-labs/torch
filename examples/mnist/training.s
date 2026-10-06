package sputniktorch.mnist.training

import sputniktorch
from sputniktorch.backends import execution_batch

export train_epoch, evaluate

# The optional profiler is used only by bench/mnist; showcase keeps its usual loop.
def train_epoch(model, batches, loss_fn, optimizer, device:, profile: null):
  execution_batch:
    _train_epoch(model, batches, loss_fn, optimizer, device:, profile:)

def _train_epoch(model, batches, loss_fn, optimizer, device:, profile:):
  model.train!(enabled: true)
  total_loss = 0.0
  seen = 0
  profile.start! if profile != null
  batches.each |(features, labels)|:
    profile.mark!(:batch) if profile != null
    features = features.to(device:)
    labels = labels.to(device:)
    profile.mark!(:transfer) if profile != null

    optimizer.zero_grad!
    profile.mark!(:zero_grad) if profile != null
    loss = loss_fn(model.call!(features), labels)
    profile.mark!(:forward) if profile != null
    loss.backward!
    profile.mark!(:backward) if profile != null
    optimizer.step!
    profile.mark!(:optimizer) if profile != null

    total_loss += loss.item * labels.numel
    seen += labels.numel
    profile.mark!(:metrics) if profile != null
  total_loss / seen

def evaluate(model, batches, loss_fn, device:):
  execution_batch:
    _evaluate(model, batches, loss_fn, device:)

def _evaluate(model, batches, loss_fn, device:):
  model.eval!
  sputniktorch.no_grad:
    total_loss = 0.0
    correct = 0
    seen = 0
    batches.each |(features, labels)|:
      features = features.to(device:)
      labels = labels.to(device:)
      logits = model(features)
      total_loss += loss_fn(logits, labels).item * labels.numel
      correct += logits.argmax(dim: 1).eq(labels).sum.item
      seen += labels.numel
    metrics = (total_loss / seen, correct.float / seen)
    metrics
