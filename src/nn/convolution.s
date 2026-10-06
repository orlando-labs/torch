package sputniktorch.nn.convolution

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import kernel_cores, spatial
from sputniktorch.tensor.factories import randn, zeros

export Convolution

class Convolution < Module:
  attr weight
  attr bias

  def _initialize_convolution!(dimensions, in_channels, out_channels, kernel_size,
      stride:, padding:, dilation:, groups:, bias:, seed:, device:,
      transposed: false, output_padding: 0):
    _initialize_registry!
    if in_channels <= 0 or out_channels <= 0 or groups <= 0:
      raise ValueError("channels and groups must be positive")
    if in_channels % groups != 0 or out_channels % groups != 0:
      raise ValueError("groups must divide input and output channels")
    sizes = spatial(kernel_size, dimensions)
    invalid = sizes.any?: $it <= 0
    if invalid:
      raise ValueError("kernel dimensions must be positive")
    @operation = "conv#{if transposed then "_transpose" else ""}#{dimensions}d"
    @options = [spatial(stride, dimensions), spatial(padding, dimensions)]
    if transposed:
      @options.push!(spatial(output_padding, dimensions))
      @options.push!(groups)
      @options.push!(spatial(dilation, dimensions))
    else:
      @options.push!(spatial(dilation, dimensions))
      @options.push!(groups)
    shape = if transposed:
      [in_channels, out_channels // groups, *sizes]
    else:
      [out_channels, in_channels // groups, *sizes]
    scale = ((in_channels // groups) * sizes.product).float ** 0.5
    @weight = (randn(shape, seed:, device:) / scale).detach.requires_grad!(true)
    @bias = if bias then zeros([out_channels], device:, requires_grad: true) else null
    @bias_core = if bias then @bias._core else null
    register_parameter!("weight", @weight)
    register_parameter!("bias", @bias) if bias
    self

  def forward(input): kernel_cores(@operation, [input._core, @weight._core, @bias_core], @options)
