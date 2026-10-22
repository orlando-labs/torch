package torch.nn.layer_norm

from torch.nn.module import Module
from torch.nn.functional import layer_norm
from torch.tensor.factories import ones, zeros

export LayerNorm

class LayerNorm < Module:
  attr weight
  attr bias

  def init(normalized_shape, @eps: 1e-5, affine: true, bias: true, device: "cpu", dtype: "float32"):
    _initialize_registry!
    @shape = if Int === normalized_shape then [normalized_shape] else normalized_shape.array
    @weight = if affine then ones(@shape, device:, dtype:, requires_grad: true) else null
    @bias = if affine and bias then zeros(@shape, device:, dtype:, requires_grad: true) else null
    register_parameter!("weight", @weight) if affine
    register_parameter!("bias", @bias) if @bias != null

  def forward(x): layer_norm(x, @shape, weight: @weight, bias: @bias, eps: @eps)
