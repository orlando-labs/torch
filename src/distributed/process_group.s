package torch.distributed.process_group

from torch.distributed.core import group_env as _group_env, group_open as _group_open
from torch.distributed.core import collective! as _collective!, gloo_available?

export ProcessGroup, open_group, group_from_env, gloo_available?

def group_from_env(timeout_ms: 20000): ProcessGroup(_group_env(timeout_ms))

def open_group(rank:, world_size:, path:, timeout_ms: 20000, host: "127.0.0.1"):
  ProcessGroup(_group_open(rank, world_size, path, timeout_ms, host))

class ProcessGroup:
  def init(@core)
  attr _core from @core
  def rank(): _collective!(@core, "rank", null, 0)
  def world_size(): _collective!(@core, "world_size", null, 0)

  def all_reduce!(tensor):
    _collective!(@core, "all_reduce", tensor._core, 0)
    tensor

  def broadcast!(tensor, src: 0):
    _collective!(@core, "broadcast", tensor._core, src)
    tensor

  def barrier!():
    _collective!(@core, "barrier", null, 0)
    self

  def close!():
    @core.destroy!
    self
