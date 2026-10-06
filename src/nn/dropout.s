package sputniktorch.nn.dropout

from sputniktorch.nn.module import Module
from sputniktorch.random.generator import Generator

export Dropout

class Dropout < Module:
  def init(@p: 0.5, seed: 0):
    _initialize_registry!
    raise ValueError("dropout p must be in 0..1") if p < 0 or p > 1
    @rng = Generator(seed:)

  def stateful?(): training? and @p > 0 and @p < 1

  def forward(x):
    raise ValueError("training dropout consumes RNG; use call! or eval!") if stateful?
    if training? and @p == 1 then x * 0 else x

  def forward!(x):
    return forward(x) unless stateful?
    mask = @rng.rand!(x.shape, device: x.device).ge(@p)
    x * mask / (1 - @p)

  def rng_state(): @rng.state

  def load_rng_state!(state):
    @rng.load_state!(state)
    self
