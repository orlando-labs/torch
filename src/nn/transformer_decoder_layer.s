package torch.nn.transformer_decoder_layer

from torch.nn.module import Module
from torch.nn.multihead_attention import MultiheadAttention
from torch.nn.linear import Linear
from torch.nn.layer_norm import LayerNorm
from torch.nn.gelu import GELU
from torch.nn.relu import ReLU
from torch.nn.dropout import Dropout

export TransformerDecoderLayer

class TransformerDecoderLayer < Module:
  def init(d_model, nhead, dim_feedforward: 2048, dropout: 0.1,
      @norm_first: false, batch_first: true, bias: true, layer_norm_eps: 1e-5,
      activation: :gelu, seed: 0, device: "cpu", dtype: "float32"):
    _initialize_registry!
    raise ValueError("activation must be :gelu or :relu") unless activation.str in ["gelu", "relu"]
    @attention = MultiheadAttention(d_model, nhead, dropout:, batch_first:, bias:, seed:, device:, dtype:)
    @cross_attention = MultiheadAttention(d_model, nhead, dropout:, batch_first:, bias:, seed: seed + 4, device:, dtype:)
    @expand = Linear(d_model, dim_feedforward, bias:, seed: seed + 8, device:, dtype:)
    @contract = Linear(dim_feedforward, d_model, bias:, seed: seed + 9, device:, dtype:)
    @activation = if activation.str == "gelu" then GELU() else ReLU()
    @norms = Array.of(3): LayerNorm(d_model, eps: layer_norm_eps, bias:, device:, dtype:)
    @drops = Array.of(4) |index|: Dropout(p: dropout, seed: seed + 10 + index)
    @norms.each_with_index |layer, index|: register_module!("norm#{index}", layer)
    @drops.each_with_index |layer, index|: register_module!("drop#{index}", layer)

  def _drop(index, x, mutate):
    if mutate then @drops[index].call!(x) else @drops[index](x)

  def _forward(x, memory, mask, memory_mask, attn_mask, memory_attn_mask,
      key_padding_mask, memory_key_padding_mask, causal, memory_causal, mutate):
    input = if @norm_first then @norms[0](x) else x
    attended = if mutate:
      @attention.call!(input, mask:, attn_mask:, key_padding_mask:, causal:)
    else:
      @attention(input, mask:, attn_mask:, key_padding_mask:, causal:)
    x = x + _drop(0, attended, mutate)
    x = @norms[0](x) unless @norm_first

    input = if @norm_first then @norms[1](x) else x
    attended = if mutate:
      @cross_attention.call!(input, key: memory, value: memory, mask: memory_mask,
        attn_mask: memory_attn_mask, key_padding_mask: memory_key_padding_mask, causal: memory_causal)
    else:
      @cross_attention(input, key: memory, value: memory, mask: memory_mask,
        attn_mask: memory_attn_mask, key_padding_mask: memory_key_padding_mask, causal: memory_causal)
    x = x + _drop(1, attended, mutate)
    x = @norms[1](x) unless @norm_first

    input = if @norm_first then @norms[2](x) else x
    hidden = _drop(2, @activation(@expand(input)), mutate)
    x = x + _drop(3, @contract(hidden), mutate)
    if @norm_first then x else @norms[2](x)

  def forward(x, memory, mask: null, memory_mask: null, attn_mask: null, memory_attn_mask: null,
      key_padding_mask: null, memory_key_padding_mask: null, causal: true, memory_causal: false):
    _forward(x, memory, mask, memory_mask, attn_mask, memory_attn_mask,
      key_padding_mask, memory_key_padding_mask, causal, memory_causal, false)

  def forward!(x, memory, mask: null, memory_mask: null, attn_mask: null, memory_attn_mask: null,
      key_padding_mask: null, memory_key_padding_mask: null, causal: true, memory_causal: false):
    _forward(x, memory, mask, memory_mask, attn_mask, memory_attn_mask,
      key_padding_mask, memory_key_padding_mask, causal, memory_causal, true)
