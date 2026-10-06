package sputniktorch.optim.sgd

from sputniktorch.optim.optimizer import Optimizer, validate_coefficients
from sputniktorch.autograd import no_grad

export SGD

class SGD < Optimizer:
  def init(parameters, lr: 0.01, @momentum: 0.0, @dampening: 0.0,
      @weight_decay: 0.0, @nesterov: false):
    if momentum < 0 or dampening < 0 or weight_decay < 0:
      raise ValueError("optimizer coefficients must be nonnegative")
    if nesterov and (momentum <= 0 or dampening != 0):
      raise ValueError("Nesterov needs positive momentum and zero dampening")
    validate_coefficients([momentum, dampening, weight_decay])
    _initialize_optimizer!(parameters, lr)
    @velocity = Array.filled(@parameters.count, null)

  def step!():
    _validate_step
    no_grad():
      @parameters.each_with_index |p, index|:
        gradient = p.grad
        if gradient == null:
          next null
        gradient = gradient + p * @weight_decay if @weight_decay != 0
        if @momentum != 0:
          velocity = @velocity[index]
          if velocity == null:
            velocity = gradient.copy
          else:
            velocity = velocity * @momentum + gradient * (1 - @dampening)
          @velocity[index] = velocity
          gradient = if @nesterov then gradient + velocity * @momentum else velocity
        p.add!(gradient * -@lr)
    self

  def state_dict():
    velocity = @velocity.map: if $it == null then null else $it.snapshot
    {kind: "SGD", lr: @lr, momentum: @momentum, dampening: @dampening,
      weight_decay: @weight_decay, nesterov: @nesterov,
      velocity:}

  def load_state_dict!(state):
    raise ValueError("optimizer kind mismatch") if state[:kind] != "SGD"
    if state[:momentum] < 0 or state[:dampening] < 0 or state[:weight_decay] < 0:
      raise ValueError("invalid optimizer coefficients")
    if state[:nesterov] and (state[:momentum] <= 0 or state[:dampening] != 0):
      raise ValueError("invalid Nesterov state")
    _validate_slots(state[:velocity])
    validate_coefficients([state[:momentum], state[:dampening], state[:weight_decay]])
    velocity = Array.of(@parameters.count) |index|:
      slot = state[:velocity][index]
      if slot == null then null else slot.to(device: @parameters[index].device).copy
    lr!(state[:lr])
    @momentum = state[:momentum]
    @dampening = state[:dampening]
    @weight_decay = state[:weight_decay]
    @nesterov = state[:nesterov]
    @velocity = velocity
    self
