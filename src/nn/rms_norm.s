package torch.nn.rms_norm

from torch.nn.module import Module
from torch.nn.functional import rms_norm
from torch.tensor.factories import ones

export RMSNorm

class RMSNorm < Module:
  attr weight

  def init(normalized_shape, @eps: 1e-5, affine: true, device: "cpu"):
    _initialize_registry!
    @shape = if Int === normalized_shape then [normalized_shape] else normalized_shape.array
    @weight = if affine then ones(@shape, device:, requires_grad: true) else null
    register_parameter!("weight", @weight) if affine

  def forward(x): rms_norm(x, @shape, weight: @weight, eps: @eps)
