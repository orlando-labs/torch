package sputniktorch.data.random_sampler

from sputniktorch.random.generator import Generator

export RandomSampler

# Epoch is explicit: repeated reads of the same epoch return the same order.
class RandomSampler:
  def init(@count, @seed: 0): @epoch = 0

  def set_epoch!(epoch):
    raise ValueError("epoch must be nonnegative") if epoch < 0
    @epoch = epoch
    self

  def indices(): Generator(seed: @seed + @epoch).randperm!(@count).to_a
