package torch.tensor.slice

export Slice

class Slice:
  def init(@start: null, @stop: null, @step: 1):
    raise ValueError("slice step must be a positive Int") unless Int === @step and @step > 0

  attr start from @start
  attr stop from @stop
  attr step from @step
  def _arguments(): [@start, @stop, @step]
