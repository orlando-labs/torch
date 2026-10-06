package sputniktorch.optim.lr_scheduler.reduce_lr_on_plateau

export ReduceLROnPlateau

class ReduceLROnPlateau:
  def init(@optimizer, @mode: :min, @factor: 0.1, @patience: 10,
      @threshold: 1e-4, @min_lr: 0.0):
    if not (mode in [:min, :max]) or factor <= 0 or factor >= 1 or patience < 0 or min_lr < 0:
      raise ValueError("invalid plateau schedule")
    @best = null
    @bad_epochs = 0

  def step!(metric):
    improved = if @best == null:
      true
    else if @mode == :min:
      metric < @best - @threshold
    else:
      metric > @best + @threshold
    if improved:
      @best = metric
      @bad_epochs = 0
    else:
      @bad_epochs += 1
      if @bad_epochs > @patience:
        @optimizer.lr!(Math.max(@min_lr, @optimizer.lr * @factor))
        @bad_epochs = 0
    self

  def state_dict(): {best: @best, bad_epochs: @bad_epochs, lr: @optimizer.lr}

  def load_state_dict!(state):
    @best = state[:best]
    @bad_epochs = state[:bad_epochs]
    @optimizer.lr!(state[:lr])
    self
