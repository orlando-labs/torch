package sputniktorch.backends

from sputniktorch.errors import Error

export version, mps_built?, mps_available?, synchronize, num_threads, num_interop_threads, execution_batch

native def _info(key as Str) from "amt_info"
native def _execution_batch(block) from "amt_execution_batch"

# Execute the Sputnik callback on one blocking FFI worker in native programs.
# Nested tensor calls reuse that worker. Task locals and faults follow the block.
def execution_batch(&block): _execution_batch(block)

def version(): _info("version")
def mps_built?(): _info("mps_built")
def mps_available?(): _info("mps_available")
def synchronize(): _info("synchronize")

def num_threads(): _info("num_threads")
def num_interop_threads(): _info("num_interop_threads")
