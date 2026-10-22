package torch.tensor.indexing_core

from torch.autograd.context import execution_mode

export indexing

native def _indexing(op as Str, inputs, options, mutating as Bool, mode as Int) from "amt_indexing" blocking

def indexing(core, op, inputs, options, mutating: false):
  _indexing(op, [core, *inputs], options, mutating, execution_mode())
