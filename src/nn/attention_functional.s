package torch.nn.attention_functional

from torch.tensor.value import Tensor
from torch.tensor.factories import arange, tensor
from torch.autograd.context import execution_mode
from torch.nn.dropout_functional import apply_dropout, sample_dropout!, validate_probability
from torch.errors import ShapeError, DTypeError, DeviceError

export scaled_dot_product_attention, scaled_dot_product_attention!
export attention, attention_mask, combine_attention_masks, causal_mask

native def _attention(op as Str, inputs, options, mode as Int) from "amt_attention" blocking

def _kernel(op, inputs, options = []):
  handles = inputs.map: if $it == null then null else $it._core
  Tensor(_attention(op, handles, options, execution_mode()))

def _broadcast(left, right):
  count = Math.max(left.count, right.count)
  Array.of(count) |i|:
    a = if i < count - left.count then 1 else left[i - count + left.count]
    b = if i < count - right.count then 1 else right[i - count + right.count]
    raise ShapeError("attention dimensions are not broadcastable") unless a == b or a == 1 or b == 1
    if a == 1 then b else a

def attention_mask(mask, like, shape):
  return null if mask == null
  raise TypeError("attention mask must be a Tensor") unless Tensor === mask
  unless mask.dtype.bool? or mask.dtype == like.dtype:
    raise DTypeError("attention mask must be boolean or match the input floating dtype")
  raise DeviceError("attention mask must share the input device") unless mask.device.to_str == like.device.to_str
  raise ShapeError("attention mask must broadcast to the scores") unless _broadcast(mask.shape, shape) == shape
  mask

def combine_attention_masks(left, right, like):
  return right if left == null
  return left if right == null
  return left * right if left.dtype.bool? and right.dtype.bool?
  left = _kernel("mask_bias", [left, like]) if left.dtype.bool?
  right = _kernel("mask_bias", [right, like]) if right.dtype.bool?
  left + right

def causal_mask(rows, columns, device):
  arange(rows, device:).unsqueeze(-1).ge(arange(columns, device:).unsqueeze(0))

def _expand_heads(x, heads):
  return x if x.shape[-3] == heads
  repeats = heads // x.shape[-3]
  slots = Array.of(heads) |i|: i // repeats
  indices = tensor(slots, dtype: "int64", device: x.device)
  x.index_select(-3, indices)

# The pair form is shared by MHA; projections and dropout policy stay in Sputnik.
def attention(query, key, value, mask: null, causal: false, scale: null,
    enable_gqa: false, dropout_p: 0.0, dropout_mask: null, generator: null, need_weights: false):
  validate_probability(dropout_p)
  [query, key, value].each |x|:
    raise TypeError("attention inputs must be Tensors") unless Tensor === x
    raise ShapeError("attention inputs require rank >= 2") if x.rank < 2
    raise DTypeError("attention inputs must be floating") unless x.dtype.floating?
    raise DTypeError("attention input dtypes must match") unless x.dtype == query.dtype
    raise DeviceError("attention input devices must match") unless x.device.to_str == query.device.to_str
  unless query.shape[-1] > 0 and query.shape[-1] == key.shape[-1] and key.shape[-2] == value.shape[-2]:
    raise ShapeError("attention key dimensions or sequence lengths do not match")
  if scale != null:
    raise TypeError("attention scale must be numeric") unless Int === scale or Float === scale
  if enable_gqa:
    unless (query.rank >= 3 and key.rank >= 3 and value.rank >= 3 and query.shape[-3] > 0 and key.shape[-3] > 0 and
        key.shape[-3] == value.shape[-3] and query.shape[-3] % key.shape[-3] == 0):
      raise ShapeError("GQA requires matching key/value heads dividing query heads")
    key = _expand_heads(key, query.shape[-3])
    value = _expand_heads(value, query.shape[-3])
  leading = _broadcast(query.shape.first(query.rank - 2), key.shape.first(key.rank - 2))
  leading = _broadcast(leading, value.shape.first(value.rank - 2))
  shape = [*leading, query.shape[-2], key.shape[-2]]
  mask = attention_mask(mask, query, shape)
  if causal:
    mask = combine_attention_masks(mask, causal_mask(query.shape[-2], key.shape[-2], query.device), query)
  if dropout_p > 0 and dropout_p < 1 and generator == null and dropout_mask == null:
    raise ValueError("attention dropout requires dropout_mask: or a bang call with Generator")
  if not need_weights and dropout_p == 0 and execution_mode() // 4 == 0:
    return (_kernel("sdpa", [query, key, value, mask], [scale]), null)
  factor = if scale == null then query.shape[-1].float ** -0.5 else scale
  scores = query.matmul(key.transpose(-1, -2)) * factor
  scores = if mask == null then scores else if mask.dtype.bool? then scores + _kernel("mask_bias", [mask, query]) else scores + mask
  probabilities = _kernel("softmax", [scores])
  probabilities = if generator == null:
    apply_dropout(probabilities, dropout_p, true, :element, mask: dropout_mask)
  else:
    sample_dropout!(probabilities, generator, dropout_p, true, :element)
  return (probabilities.matmul(value), probabilities)

def _options(mask, attn_mask, causal, is_causal):
  raise ValueError("provide mask: or attn_mask:, not both") if mask != null and attn_mask != null
  raise ValueError("causal: and is_causal: disagree") if causal and is_causal == false
  selected = if attn_mask == null then mask else attn_mask
  triangular = if is_causal == null then causal else is_causal
  return (selected, triangular)

def scaled_dot_product_attention(query, key, value, mask: null, attn_mask: null,
    causal: false, is_causal: null, scale: null, enable_gqa: false, dropout_p: 0.0, dropout_mask: null):
  (mask, causal) = _options(mask, attn_mask, causal, is_causal)
  attention(query, key, value, mask:, causal:, scale:, enable_gqa:, dropout_p:, dropout_mask:)[0]

def scaled_dot_product_attention!(query, key, value, generator, mask: null, attn_mask: null,
    causal: false, is_causal: null, scale: null, enable_gqa: false, dropout_p: 0.0):
  (mask, causal) = _options(mask, attn_mask, causal, is_causal)
  attention(query, key, value, mask:, causal:, scale:, enable_gqa:, dropout_p:, generator:)[0]
