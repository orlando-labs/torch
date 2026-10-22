package torch.nn.instance_norm

from torch.nn.batch_norm import BatchNorm
from torch.nn.functional import instance_norm, instance_norm!

export InstanceNorm

class InstanceNorm < BatchNorm:
  def _check_input(x):
    unless x.rank in [@dimensions + 1, @dimensions + 2]:
      raise ValueError("wrong input rank for InstanceNorm#{@dimensions}d")
    channel_axis = if x.rank == @dimensions + 1 then 0 else 1
    if (@affine or @tracking) and x.shape[channel_axis] != @features:
      raise ValueError("input channels must match features")

  def forward(x):
    raise ValueError("training InstanceNorm changes buffers; use call! or eval!") if stateful?
    _apply(x, false)

  def forward!(x): _apply(x, stateful?)

  def _apply(x, mutating):
    _check_input(x)
    unbatched = x.rank == @dimensions + 1
    input = if unbatched then x.unsqueeze(0) else x
    momentum = if @momentum == null then 0.0 else @momentum
    result = if mutating:
      instance_norm!(input, running_mean: @running_mean, running_var: @running_var,
        weight: @weight, bias: @bias, momentum:, eps: @eps)
    else:
      instance_norm(input, running_mean: @running_mean, running_var: @running_var,
        weight: @weight, bias: @bias, use_input_stats: training? or not @tracking,
        momentum:, eps: @eps)
    if unbatched then result.squeeze(dim: 0) else result
