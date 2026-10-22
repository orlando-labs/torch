package torch.nn.group_norm

from torch.nn.module import Module
from torch.nn.functional import group_norm
from torch.tensor.factories import ones, zeros

export GroupNorm

class GroupNorm < Module:
  attr weight
  attr bias

  def init(@groups, channels, @eps: 1e-5, affine: true, device: "cpu"):
    _initialize_registry!
    if groups <= 0 or channels <= 0 or channels % groups != 0:
      raise ValueError("groups must divide channels")
    @weight = if affine then ones([channels], device:, requires_grad: true) else null
    @bias = if affine then zeros([channels], device:, requires_grad: true) else null
    register_parameter!("weight", @weight) if affine
    register_parameter!("bias", @bias) if affine

  def forward(x): group_norm(x, @groups, weight: @weight, bias: @bias, eps: @eps)
