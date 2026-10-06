package sputniktorch.nn.multihead_attention

from sputniktorch.nn.module import Module
from sputniktorch.nn.linear import Linear
from sputniktorch.nn.functional import scaled_dot_product_attention

export MultiheadAttention

# Batch-first [batch, sequence, embedding]; weights remain inspectable Sputnik
# submodules. Projection, head layout and attention composition are explicit.
class MultiheadAttention < Module:
  attr query_projection
  attr key_projection
  attr value_projection
  attr output_projection

  def init(@embed_dim, @num_heads, seed: 0, device: "cpu"):
    _initialize_registry!
    if num_heads <= 0 or embed_dim <= 0 or embed_dim % num_heads != 0:
      raise ValueError("num_heads must divide embed_dim")
    @query_projection = Linear(embed_dim, embed_dim, seed:, device:)
    @key_projection = Linear(embed_dim, embed_dim, seed: seed + 1, device:)
    @value_projection = Linear(embed_dim, embed_dim, seed: seed + 2, device:)
    @output_projection = Linear(embed_dim, embed_dim, seed: seed + 3, device:)
    register_module!("query", @query_projection)
    register_module!("key", @key_projection)
    register_module!("value", @value_projection)
    register_module!("output", @output_projection)

  def _heads(x):
    x.reshape([x.shape[0], x.shape[1], @num_heads, @embed_dim // @num_heads]).transpose(1, 2)

  def forward(query, key: null, value: null, mask: null, causal: false):
    source = if key == null then query else key
    values = if value == null then source else value
    q = _heads(@query_projection(query))
    k = _heads(@key_projection(source))
    v = _heads(@value_projection(values))
    output = scaled_dot_product_attention(q, k, v, mask:, causal:)
    @output_projection(output.transpose(1, 2).reshape(query.shape))
