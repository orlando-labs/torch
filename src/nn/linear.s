package torch.nn.linear

from torch.nn.module import Module
from torch.tensor.factories import randn, zeros

export Linear

class Linear < Module:
  attr weight
  attr bias

  def init(in_features, out_features, bias: true, seed: 0, device: "cpu", dtype: "float32"):
    _initialize_registry!
    raise ValueError("feature counts must be positive") if in_features <= 0 or out_features <= 0
    @weight = (randn([out_features, in_features], seed:, device:, dtype:) * 0.1).detach.requires_grad!(true)
    @bias = if bias then zeros([out_features], device:, dtype:, requires_grad: true) else null
    register_parameter!("weight", @weight)
    register_parameter!("bias", @bias) if bias

  def forward(x):
    output = x.matmul(@weight.transpose(0, 1))
    if @bias == null then output else output + @bias
