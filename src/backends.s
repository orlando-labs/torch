package torch.backends

from torch.errors import Error
from torch.autograd.context import execution_mode

export version, mps_built?, mps_available?, cuda_built?, cuda_available?, cuda_device_count, synchronize, num_threads, num_interop_threads, execution_batch

native def _info(key as Str) from "amt_info"
native def _synchronize() from "amt_synchronize" blocking
native def _execution_batch(block, mode as Int) from "amt_execution_batch" blocking

# Execute the Sputnik callback on one blocking FFI worker in native programs.
# Nested tensor calls reuse that worker. Task locals and faults follow the block.
def execution_batch(&block): _execution_batch(block, execution_mode())

def version(): _info("version")
def mps_built?(): _info("mps_built")
def mps_available?(): _info("mps_available")
def cuda_built?(): _info("cuda_built")
def cuda_available?(): _info("cuda_available")
def cuda_device_count(): _info("cuda_device_count")
def synchronize(): _synchronize()

def num_threads(): _info("num_threads")
def num_interop_threads(): _info("num_interop_threads")
