package torch.distributed.core

from torch.errors import Error

export GroupCore, group_env, group_open, collective!, gloo_available?

native class GroupCore from "torch.GroupCore" owned:
  def destroy!() from "amt_group_free"

native def group_env(timeout_ms) from "amt_group_env" blocking
native def group_open(rank, world_size, path, timeout_ms, host) from "amt_group_open" blocking
native def collective!(group, op, tensor, root) from "amt_collective" blocking
native def gloo_available?() from "amt_gloo_available"
