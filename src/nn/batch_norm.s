package torch.nn.batch_norm

from torch.nn.module import Module
from torch.nn.functional import batch_norm, batch_norm!
from torch.tensor.factories import ones, zeros
from torch.autograd.context import no_grad

export BatchNorm

class BatchNorm < Module:
  attr weight
  attr bias
  attr running_mean
  attr running_var
  attr num_batches_tracked
  attr features from @features
  attr affine? from @affine
  attr track_running_stats? from @tracking

  def _initialize_batch_norm!(dimensions, features, eps:, momentum:, affine:,
      track_running_stats:, device:, dtype: "float32", bias: true):
    _initialize_registry!
    raise ValueError("features must be a positive Int") unless Int === features and features > 0
    @dimensions = dimensions
    @features = features
    @affine = affine
    @eps = eps
    @momentum = momentum
    @tracking = track_running_stats
    @weight = if affine then ones([features], device:, dtype:, requires_grad: true) else null
    @bias = if affine and bias then zeros([features], device:, dtype:, requires_grad: true) else null
    register_parameter!("weight", @weight) if @weight != null
    register_parameter!("bias", @bias) if @bias != null
    @running_mean = if @tracking then zeros([features], device:, dtype:) else null
    @running_var = if @tracking then ones([features], device:, dtype:) else null
    @num_batches_tracked = if @tracking then zeros([], dtype: "int64", device:) else null
    if @tracking:
      register_buffer!("running_mean", @running_mean)
      register_buffer!("running_var", @running_var)
      register_buffer!("num_batches_tracked", @num_batches_tracked)
    self

  def stateful?(): training? and @tracking

  def _check_input(x):
    valid = if @dimensions == 1 then x.rank in [2, 3] else x.rank == @dimensions + 2
    raise ValueError("wrong input rank for BatchNorm#{@dimensions}d") unless valid
    if (@affine or @tracking) and x.shape[1] != @features:
      raise ValueError("input channels must match features")

  def forward(x):
    if stateful?:
      raise ValueError("training BatchNorm changes buffers; use forward! or eval!")
    _check_input(x)
    momentum = if @momentum == null then 0.0 else @momentum
    batch_norm(x, @running_mean, @running_var, weight: @weight, bias: @bias,
      training: not @tracking, momentum:, eps: @eps)

  def forward!(x):
    return forward(x) unless stateful?
    _check_input(x)
    momentum = if @momentum == null then 1.0 / (@num_batches_tracked.item + 1) else @momentum
    result = batch_norm!(x, @running_mean, @running_var, weight: @weight,
      bias: @bias, momentum:, eps: @eps)
    @num_batches_tracked.add!(1)
    result

  def reset_running_stats!():
    if @tracking:
      @running_mean.zero!
      @running_var.fill!(1)
      @num_batches_tracked.zero!
    self

  def reset_parameters!():
    reset_running_stats!
    no_grad():
      @weight.fill!(1) if @weight != null
      @bias.zero! if @bias != null
    self
