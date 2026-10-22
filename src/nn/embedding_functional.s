package torch.nn.embedding_functional

from torch.tensor.value import Tensor
from torch.tensor.factories import arange, randn, zeros, ones, full_like, ones_like
from torch.autograd.context import no_grad, execution_mode
from torch.linalg import vector_norm
from torch.errors import ReadOnlyTensorError

export embedding, embedding!, embedding_bag, embedding_bag!
export embedding_renorm!, embedding_renormed, padding_index, embedding_weight, bag_mode

native def _embedding(op as Str, inputs, options, mode as Int) from "amt_embedding" blocking

def _dispatch(op, inputs, options = []):
  handles = inputs.map: if $it == null then null else $it._core
  _embedding(op, handles, options, execution_mode())

def padding_index(index, vocabulary):
  return null if index == null
  unless Int === index and index >= -vocabulary and index < vocabulary:
    raise ValueError("padding_idx is outside the vocabulary")
  if index < 0 then vocabulary + index else index

def embedding_weight(vocabulary, dimensions, padding, initial, freeze, seed, device, dtype):
  unless Int === vocabulary and Int === dimensions and vocabulary > 0 and dimensions > 0:
    raise ValueError("embedding dimensions must be positive Ints")
  weight = null
  if initial == null:
    weight = randn([vocabulary, dimensions], seed:, device:, dtype:)
    if padding != null:
      no_grad(): weight[padding].zero!
  else:
    unless Tensor === initial and initial.rank == 2 and initial.shape == [vocabulary, dimensions] and initial.dtype.floating?:
      raise ValueError("initial embedding weight must be a matching floating matrix")
    weight = initial.detach.copy
  weight.requires_grad!(not freeze)

def bag_mode(mode, scale_grad_by_freq, sparse):
  code = case! mode.str:
    when "sum": 0
    when "mean": 1
    when "max": 2
    else: raise ValueError("embedding bag mode must be :sum, :mean or :max")
  if code == 2 and (scale_grad_by_freq or sparse):
    raise ValueError("max embedding bags do not support frequency scaling or sparse gradients")
  code

def embedding(input, weight, padding_idx: null, max_norm: null, norm_type: 2.0,
    scale_grad_by_freq: false, sparse: false):
  raise ValueError("max_norm changes weight; use embedding!") if max_norm != null
  padding = padding_index(padding_idx, weight.shape[0])
  Tensor(_dispatch("embedding", [input, weight], [padding, scale_grad_by_freq, sparse]))

def embedding_renorm!(weight, input, max_norm, norm_type: 2.0):
  _dispatch("validate", [input, weight])
  raise ReadOnlyTensorError("embedding renorm needs writable weight") if weight.readonly?
  return weight if input.numel == 0
  no_grad():
    indices = input.flatten.to(device: input.device, dtype: "int64")
    rows = weight.index_select(0, indices)
    norms = vector_norm(rows, ord: norm_type, dim: 1, keepdim: true)
    factors = full_like(norms, max_norm) / (norms + 1e-7)
    adjusted = Tensor(_dispatch("where", [norms.gt(max_norm), factors, ones_like(norms)]))
    # Duplicate indices write the same normalized row; untouched rows stay intact.
    weight.index_copy!(0, indices, rows * adjusted)
  weight

def embedding_renormed(weight, input, max_norm, norm_type: 2.0):
  result = weight.copy
  embedding_renorm!(result, input, max_norm, norm_type:)

def embedding!(input, weight, padding_idx: null, max_norm: null, norm_type: 2.0,
    scale_grad_by_freq: false, sparse: false):
  padding = padding_index(padding_idx, weight.shape[0])
  options = [padding, scale_grad_by_freq, sparse]
  _dispatch("embedding_validate", [input, weight], options)
  embedding_renorm!(weight, input, max_norm, norm_type:) if max_norm != null
  Tensor(_dispatch("embedding", [input, weight], options))

def _bag_inputs(input, offsets, per_sample_weights, include_last_offset):
  if per_sample_weights != null and per_sample_weights.shape != input.shape:
    raise ValueError("per_sample_weights must match the input shape")
  if input.rank == 2:
    raise ValueError("2D embedding bags do not accept offsets") if offsets != null
    offsets = arange(input.shape[0], dtype: input.dtype, device: input.device) * input.shape[1]
    input = input.flatten
    per_sample_weights = per_sample_weights.flatten if per_sample_weights != null
    include_last_offset = false
  elif input.rank != 1 or offsets == null:
    raise ValueError("embedding bags require 2D input or 1D input with offsets")
  return (input, offsets, per_sample_weights, include_last_offset)

def _bag_options(weight, mode, padding_idx, scale_grad_by_freq, sparse, include_last_offset, per_sample_weights):
  code = bag_mode(mode, scale_grad_by_freq, sparse)
  if per_sample_weights != null and code != 0:
    raise ValueError("per_sample_weights is supported only for sum bags")
  [code, scale_grad_by_freq, sparse, include_last_offset, padding_index(padding_idx, weight.shape[0])]

def _bag_result(input, weight, offsets, per_sample_weights, options):
  if weight.device.type == "mps" and options[1]:
    _dispatch("bag_validate", [input, weight, offsets, per_sample_weights], options)
    counts = zeros([weight.shape[0], 1], dtype: weight.dtype, device: weight.device)
    no_grad():
      counts.index_add!(0, input.to(device: input.device, dtype: "int64"),
        ones([input.numel, 1], dtype: weight.dtype, device: weight.device))
    # Keep lookup values unchanged while scaling only the weight derivative.
    fixed = weight.detach
    weight = fixed + (weight - fixed) / counts.clamp(min: 1, max: input.numel + 1)
    options = [options[0], false, options[2], options[3], options[4]]
  Tensor(_dispatch("bag", [input, weight, offsets, per_sample_weights], options))

def embedding_bag(input, weight, offsets = null, mode: :mean, padding_idx: null,
    max_norm: null, norm_type: 2.0, scale_grad_by_freq: false, sparse: false,
    per_sample_weights: null, include_last_offset: false):
  raise ValueError("max_norm changes weight; use embedding_bag!") if max_norm != null
  prepared = _bag_inputs(input, offsets, per_sample_weights, include_last_offset)
  (input, offsets, per_sample_weights, include_last_offset) = prepared
  options = _bag_options(weight, mode, padding_idx, scale_grad_by_freq, sparse, include_last_offset, per_sample_weights)
  _bag_result(input, weight, offsets, per_sample_weights, options)

def embedding_bag!(input, weight, offsets = null, mode: :mean, padding_idx: null,
    max_norm: null, norm_type: 2.0, scale_grad_by_freq: false, sparse: false,
    per_sample_weights: null, include_last_offset: false):
  prepared = _bag_inputs(input, offsets, per_sample_weights, include_last_offset)
  (input, offsets, per_sample_weights, include_last_offset) = prepared
  options = _bag_options(weight, mode, padding_idx, scale_grad_by_freq, sparse, include_last_offset, per_sample_weights)
  _dispatch("bag_validate", [input, weight, offsets, per_sample_weights], options)
  embedding_renorm!(weight, input, max_norm, norm_type:) if max_norm != null
  _bag_result(input, weight, offsets, per_sample_weights, options)
