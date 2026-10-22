package torch.tensor.foreach

from torch.autograd.context import execution_mode

# Internal TensorCore-list boundary. Algorithms and per-parameter state stay
# in Sputnik; ATen supplies only generic list tensor primitives.
export binary, unary, mutate!, gradients, zero_grad!, packable?, independent?, pack, unpack, index_select

native def _binary(cores, other, op as Str, mode as Int) from "amt_foreach_binary"
native def _unary(cores, op as Str, mode as Int) from "amt_foreach_unary"
native def _mutate!(cores, other, op as Str, mode as Int) from "amt_foreach_mutate"
native def _gradients(cores, mode as Int) from "amt_foreach_gradients"
native def _zero_grad!(cores, release as Bool, mode as Int) from "amt_foreach_zero_grad"
native def packable?(cores) from "amt_tensor_list_packable"
native def independent?(cores) from "amt_tensor_list_independent"
native def _pack(cores, mode as Int) from "amt_tensor_list_pack"
native def _unpack(core, templates, mode as Int) from "amt_tensor_list_unpack"
native def _index_select(cores, dim as Int, indices, mode as Int) from "amt_tensor_list_index_select"

def binary(cores, other, op): _binary(cores, other, op, execution_mode())
def unary(cores, op): _unary(cores, op, execution_mode())
def mutate!(cores, other, op): _mutate!(cores, other, op, execution_mode())
def gradients(cores): _gradients(cores, execution_mode())
def zero_grad!(cores, release): _zero_grad!(cores, release, execution_mode())
def pack(cores): _pack(cores, execution_mode())
def unpack(core, templates): _unpack(core, templates, execution_mode())
def index_select(cores, dim, indices): _index_select(cores, dim, indices, execution_mode())
