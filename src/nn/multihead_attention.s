package torch.nn.multihead_attention

from torch.nn.module import Module
from torch.nn.linear import Linear
from torch.nn.functional import cat
from torch.nn.attention_functional import attention, attention_mask, combine_attention_masks, causal_mask
from torch.tensor.value import Tensor
from torch.tensor.factories import randn, ones, zeros
from torch.nn.dropout_functional import validate_probability
from torch.random.generator import Generator
from torch.errors import ShapeError, DTypeError, DeviceError

export MultiheadAttention

# Existing batch-first and Tensor-output defaults remain compatible.
class MultiheadAttention < Module:
  attr query_projection
  attr key_projection
  attr value_projection
  attr output_projection
  attr bias_k
  attr bias_v
  attr batch_first
  attr embed_dim
  attr num_heads
  attr kdim
  attr vdim

  def init(@embed_dim, @num_heads, dropout: 0.0, bias: true, add_bias_kv: false,
      @add_zero_attn: false, kdim: null, vdim: null, @batch_first: true,
      seed: 0, device: "cpu", dtype: "float32"):
    _initialize_registry!
    unless Int === num_heads and Int === embed_dim and num_heads > 0 and embed_dim > 0 and embed_dim % num_heads == 0:
      raise ValueError("num_heads must divide embed_dim")
    @kdim = if kdim == null then embed_dim else kdim
    @vdim = if vdim == null then embed_dim else vdim
    unless Int === @kdim and Int === @vdim and @kdim > 0 and @vdim > 0:
      raise ValueError("key/value dimensions must be positive Ints")
    validate_probability(dropout)
    @dropout = dropout
    @rng = Generator(seed: seed + 6)
    @query_projection = Linear(embed_dim, embed_dim, bias:, seed:, device:, dtype:)
    @key_projection = Linear(@kdim, embed_dim, bias:, seed: seed + 1, device:, dtype:)
    @value_projection = Linear(@vdim, embed_dim, bias:, seed: seed + 2, device:, dtype:)
    @output_projection = Linear(embed_dim, embed_dim, bias:, seed: seed + 3, device:, dtype:)
    register_module!("query", @query_projection)
    register_module!("key", @key_projection)
    register_module!("value", @value_projection)
    register_module!("output", @output_projection)
    @bias_k = null
    @bias_v = null
    if add_bias_kv:
      @bias_k = randn([1, 1, embed_dim], seed: seed + 4, device:, dtype:, requires_grad: true)
      @bias_v = randn([1, 1, embed_dim], seed: seed + 5, device:, dtype:, requires_grad: true)
      register_parameter!("bias_k", @bias_k)
      register_parameter!("bias_v", @bias_v)

  def stateful?(): training? and @dropout > 0 and @dropout < 1
  def rng_state(): @rng.state

  def load_rng_state!(state):
    @rng.load_state!(state)
    self

  def _heads(x):
    x.reshape([x.shape[0], x.shape[1], @num_heads, @embed_dim // @num_heads]).transpose(1, 2)

  def _input(x, width, query):
    raise TypeError("MHA inputs must be Tensors") unless Tensor === x
    raise ShapeError("MHA requires rank 2 or 3 and matching feature widths") unless x.rank in [2, 3] and x.shape[-1] == width
    raise ShapeError("MHA input ranks must match") unless x.rank == query.rank
    raise DTypeError("MHA input dtypes must match and be floating") unless x.dtype.floating? and x.dtype == query.dtype
    raise DeviceError("MHA inputs must share a device") unless x.device.to_str == query.device.to_str
    if x.rank == 2 then x.unsqueeze(0) else if @batch_first then x else x.transpose(0, 1)

  def _mask(mask, attn_mask, padding, query, batch, rows, columns, unbatched):
    raise ValueError("provide mask: or attn_mask:, not both") if mask != null and attn_mask != null
    shape = [batch, @num_heads, rows, columns]
    if attn_mask != null:
      raise TypeError("attn_mask must be a Tensor") unless Tensor === attn_mask
      attention_mask(attn_mask, query, attn_mask.shape)
      if attn_mask.shape == [rows, columns]:
        mask = attn_mask
      elif attn_mask.shape == [batch * @num_heads, rows, columns]:
        mask = attn_mask.reshape(shape)
      else:
        raise ShapeError("attn_mask must be [L,S] or [N*heads,L,S]")
      mask = mask.eq(false) if mask.dtype.bool?
    mask = attention_mask(mask, query, shape)
    if padding != null:
      raise TypeError("key_padding_mask must be a Tensor") unless Tensor === padding
      attention_mask(padding, query, padding.shape)
      expected = if unbatched then [columns] else [batch, columns]
      raise ShapeError("key_padding_mask has the wrong shape") unless padding.shape == expected
      padding = padding.eq(false) if padding.dtype.bool?
      padding = padding.reshape([batch, 1, 1, columns])
      mask = combine_attention_masks(mask, padding, query)
    mask

  def _pad_mask(mask):
    return null if mask == null
    shape = [*mask.shape.first(mask.rank - 1), 1]
    extra = if mask.dtype.bool? then ones(shape, dtype: "bool", device: mask.device) else zeros(shape, dtype: mask.dtype, device: mask.device)
    cat([mask, extra], dim: -1)

  def _forward(query, key, value, mask, attn_mask, key_padding_mask, causal, is_causal,
      need_weights, average_attn_weights, mutate):
    raise ValueError("attention dropout consumes RNG; use call! or eval!") if stateful? and not mutate
    source = if key == null then query else key
    values = if value == null then source else value
    q_input = _input(query, @embed_dim, query)
    k_input = _input(source, @kdim, query)
    v_input = _input(values, @vdim, query)
    batch = q_input.shape[0]
    rows = q_input.shape[1]
    columns = k_input.shape[1]
    unless k_input.shape[0] == batch and v_input.shape[0] == batch and v_input.shape[1] == columns:
      raise ShapeError("MHA batch sizes and key/value lengths must match")
    unbatched = query.rank == 2
    mask = _mask(mask, attn_mask, key_padding_mask, query, batch, rows, columns, unbatched)
    raise ValueError("causal: and is_causal: disagree") if causal and is_causal == false
    causal = is_causal unless is_causal == null
    # Mask the original sequence before appending always-visible extra slots.
    if causal:
      mask = combine_attention_masks(mask, causal_mask(rows, columns, query.device), query)
    q = _heads(@query_projection(q_input))
    k = @key_projection(k_input)
    v = @value_projection(v_input)
    if @bias_k != null:
      k = cat([k, @bias_k + zeros([batch, 1, @embed_dim], dtype: k.dtype, device: k.device)], dim: 1)
      v = cat([v, @bias_v + zeros([batch, 1, @embed_dim], dtype: v.dtype, device: v.device)], dim: 1)
      mask = _pad_mask(mask)
    if @add_zero_attn:
      extra = zeros([batch, 1, @embed_dim], dtype: k.dtype, device: k.device)
      k = cat([k, extra], dim: 1)
      v = cat([v, extra], dim: 1)
      mask = _pad_mask(mask)
    p = if training? then @dropout else 0.0
    result = attention(q, _heads(k), _heads(v), mask:, dropout_p: p,
      generator: if mutate then @rng else null, need_weights:)
    (output, weights) = result
    output = @output_projection(output.transpose(1, 2).reshape([batch, rows, @embed_dim]))
    output = if unbatched then output.squeeze(dim: 0) else if @batch_first then output else output.transpose(0, 1)
    return output unless need_weights
    weights = weights.mean(dim: 1) if average_attn_weights
    weights = weights.squeeze(dim: 0) if unbatched
    return (output, weights)

  def forward(query, key: null, value: null, mask: null, attn_mask: null,
      key_padding_mask: null, causal: false, is_causal: null, need_weights: false, average_attn_weights: true):
    _forward(query, key, value, mask, attn_mask, key_padding_mask, causal, is_causal, need_weights, average_attn_weights, false)

  def forward!(query, key: null, value: null, mask: null, attn_mask: null,
      key_padding_mask: null, causal: false, is_causal: null, need_weights: false, average_attn_weights: true):
    _forward(query, key, value, mask, attn_mask, key_padding_mask, causal, is_causal, need_weights, average_attn_weights, true)
