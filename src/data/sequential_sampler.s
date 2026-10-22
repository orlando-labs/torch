package torch.data.sequential_sampler

export SequentialSampler

class SequentialSampler:
  def init(@count)
  def indices(): (0...@count).array
