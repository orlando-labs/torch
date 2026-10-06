package sputniktorch.optim.adamw

from sputniktorch.optim.adam import Adam

export AdamW

# The decay policy is explicit; the moment algorithm stays shared in Sputnik.
class AdamW < Adam:
  def init(parameters, lr: 0.001, betas: (0.9, 0.999), eps: 1e-8, weight_decay: 0.01):
    _initialize_adam!(parameters, lr, betas, eps, weight_decay, true)
