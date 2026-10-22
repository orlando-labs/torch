package torch.nn.dropout

from torch.nn.module import Module
from torch.random.generator import Generator
from torch.nn.dropout_functional import apply_dropout, sample_dropout!, validate_probability

export Dropout

class Dropout < Module:
  def init(p: 0.5, seed: 0): _initialize_dropout!(p, seed, :element)

  def _initialize_dropout!(p, seed, kind):
    _initialize_registry!
    validate_probability(p)
    @p = p
    @kind = kind
    @rng = Generator(seed:)

  def stateful?(): training? and @p > 0 and @p < 1

  def forward(x):
    raise ValueError("training dropout consumes RNG; use call! or eval!") if stateful?
    apply_dropout(x, @p, training?, @kind)

  def forward!(x):
    sample_dropout!(x, @rng, @p, training?, @kind)

  def rng_state(): @rng.state

  def load_rng_state!(state):
    @rng.load_state!(state)
    self
