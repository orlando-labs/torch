package torch.random.core

from torch.errors import Error

export GeneratorCore, make, sample!

native class GeneratorCore from "torch.GeneratorCore" collected:
  def destroy!() from "amt_generator_free"

native def make(seed as Int) from "amt_generator_new"
native def sample!(core, operation as Str, options, dtype as Str, device as Str) from "amt_generator"
