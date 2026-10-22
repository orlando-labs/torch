package torch.optim.lr_scheduler.step_lr

export StepLR

class StepLR:
  attr steps

  def init(@optimizer, @step_size, @gamma: 0.1):
    raise ValueError("invalid StepLR configuration") if step_size <= 0 or gamma < 0
    @base_lr = optimizer.lr
    @steps = 0

  def step!():
    @steps += 1
    @optimizer.lr!(@base_lr * @gamma ** (@steps // @step_size))
    self

  def state_dict(): {steps: @steps, base_lr: @base_lr}

  def load_state_dict!(state):
    @steps = state[:steps]
    @base_lr = state[:base_lr]
    @optimizer.lr!(@base_lr * @gamma ** (@steps // @step_size))
    self
