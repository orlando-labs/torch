package sputniktorch.data.distributed_sampler

from sputniktorch.data.random_sampler import RandomSampler
from sputniktorch.random.generator import Generator

export DistributedSampler

class DistributedSampler < RandomSampler:
  def init(count, @rank:, @world_size:, seed: 0, @shuffle: true, @drop_last: false):
    if world_size <= 0 or rank < 0 or rank >= world_size:
      raise ValueError("invalid sampler rank/world_size")
    @count = count
    @seed = seed
    @epoch = 0

  def indices():
    return [] if @count == 0
    order = if @shuffle:
      Generator(seed: @seed + @epoch).randperm!(@count).to_a
    else:
      (0...@count).array
    total = if @drop_last:
      (@count // @world_size) * @world_size
    else:
      ((@count + @world_size - 1) // @world_size) * @world_size
    Array.of(total // @world_size) |i|: order[(@rank + i * @world_size) % @count]
