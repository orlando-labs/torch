package torch.optim.grad

from torch.autograd import no_grad

export clip_grad_norm!, clip_grad_value!

def clip_grad_norm!(parameters, max_norm:):
  raise ValueError("max_norm must be nonnegative") if max_norm < 0
  unique = parameters.uniq: $it._identity
  gradients = unique.filter_map: $it.grad
  norm = gradients.reduce(0.0) |total, grad|: total + grad.square.sum().item
  norm = norm ** 0.5
  coefficient = if norm > max_norm then max_norm / (norm + 1e-6) else 1.0
  no_grad(): gradients.each |grad|: grad.mul!(coefficient)
  norm

def clip_grad_value!(parameters, clip_value:):
  raise ValueError("clip_value must be nonnegative") if clip_value < 0
  unique = parameters.uniq: $it._identity
  no_grad():
    unique.each |parameter|:
      gradient = parameter.grad
      if gradient == null:
        next null
      clipped = gradient.clamp(min: -clip_value, max: clip_value)
      gradient.copy_from!(clipped)
  parameters
