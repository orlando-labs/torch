package sputniktorch.nn.batch_norm

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import batch_norm, batch_norm!
from sputniktorch.tensor.factories import ones, zeros

export BatchNorm

class BatchNorm < Module:
  attr weight
  attr bias
  attr running_mean
  attr running_var
  attr num_batches_tracked

  def _initialize_batch_norm!(dimensions, features, eps:, momentum:, affine:,
      track_running_stats:, device:):
    _initialize_registry!
    raise ValueError("features must be positive") if features <= 0
    @dimensions = dimensions
    @eps = eps
    @momentum = momentum
    @tracking = track_running_stats
    @weight = if affine then ones([features], device:, requires_grad: true) else null
    @bias = if affine then zeros([features], device:, requires_grad: true) else null
    register_parameter!("weight", @weight) if affine
    register_parameter!("bias", @bias) if affine
    @running_mean = if @tracking then zeros([features], device:) else null
    @running_var = if @tracking then ones([features], device:) else null
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

  def forward(x):
    if stateful?:
      raise ValueError("training BatchNorm changes buffers; use forward! or eval!")
    _check_input(x)
    batch_norm(x, @running_mean, @running_var, weight: @weight, bias: @bias,
      training: not @tracking, momentum: @momentum, eps: @eps)

  def forward!(x):
    return forward(x) unless stateful?
    _check_input(x)
    result = batch_norm!(x, @running_mean, @running_var, weight: @weight,
      bias: @bias, momentum: @momentum, eps: @eps)
    @num_batches_tracked.add!(1)
    result
