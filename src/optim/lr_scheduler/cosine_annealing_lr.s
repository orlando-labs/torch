package sputniktorch.optim.lr_scheduler.cosine_annealing_lr

export CosineAnnealingLR

class CosineAnnealingLR:
  attr steps

  def init(@optimizer, @t_max, @eta_min: 0.0):
    raise ValueError("invalid cosine schedule") if t_max <= 0 or eta_min < 0
    @base_lr = optimizer.lr
    @steps = 0

  def _lr(): @eta_min + (@base_lr - @eta_min) * (1 + Math.cos(Math.PI * @steps / @t_max)) / 2

  def step!():
    @steps += 1
    @optimizer.lr!(_lr)
    self

  def state_dict(): {steps: @steps, base_lr: @base_lr}

  def load_state_dict!(state):
    @steps = state[:steps]
    @base_lr = state[:base_lr]
    @optimizer.lr!(_lr)
    self
