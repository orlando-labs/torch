package sputniktorch.mnist.batch_profile

import sputniktorch

export BatchProfile, elapsed, synchronize

def elapsed(started): (Time.monotonic - started).total_nanoseconds.float / 1e9

def synchronize(device): sputniktorch.synchronize() if device == "mps"

# Synchronizing every stage changes GPU overlap. Use only for diagnosis,
# never mix these samples with the ordinary throughput measurements.
class BatchProfile:
  attr seconds

  def init(@device):
    @seconds = {batch: 0.0, transfer: 0.0, zero_grad: 0.0, forward: 0.0,
      backward: 0.0, optimizer: 0.0, metrics: 0.0}

  def start!():
    synchronize(@device)
    @started = Time.monotonic

  def mark!(key):
    synchronize(@device)
    now = Time.monotonic
    @seconds[key] = @seconds[key] + (now - @started).total_nanoseconds.float / 1e9
    @started = now
