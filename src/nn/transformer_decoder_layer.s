package sputniktorch.nn.transformer_decoder_layer

from sputniktorch.nn.module import Module
from sputniktorch.nn.multihead_attention import MultiheadAttention
from sputniktorch.nn.linear import Linear
from sputniktorch.nn.layer_norm import LayerNorm
from sputniktorch.nn.gelu import GELU
from sputniktorch.nn.dropout import Dropout

export TransformerDecoderLayer

class TransformerDecoderLayer < Module:
  def init(d_model, nhead, dim_feedforward: 2048, dropout: 0.1,
      @norm_first: false, seed: 0, device: "cpu"):
    _initialize_registry!
    @attention = MultiheadAttention(d_model, nhead, seed:, device:)
    @cross_attention = MultiheadAttention(d_model, nhead, seed: seed + 4, device:)
    @expand = Linear(d_model, dim_feedforward, seed: seed + 8, device:)
    @contract = Linear(dim_feedforward, d_model, seed: seed + 9, device:)
    @activation = GELU()
    @norms = Array.of(3): LayerNorm(d_model, device:)
    @drops = Array.of(4) |index|: Dropout(p: dropout, seed: seed + 10 + index)
    @norms.each_with_index |layer, index|: register_module!("norm#{index}", layer)
    @drops.each_with_index |layer, index|: register_module!("drop#{index}", layer)

  def _drop(index, x, mutate):
    if mutate then @drops[index].call!(x) else @drops[index](x)

  def _forward(x, memory, mask, memory_mask, causal, mutate):
    input = if @norm_first then @norms[0](x) else x
    x = x + _drop(0, @attention(input, mask:, causal:), mutate)
    x = @norms[0](x) unless @norm_first

    input = if @norm_first then @norms[1](x) else x
    attended = @cross_attention(input, key: memory, value: memory, mask: memory_mask)
    x = x + _drop(1, attended, mutate)
    x = @norms[1](x) unless @norm_first

    input = if @norm_first then @norms[2](x) else x
    hidden = _drop(2, @activation(@expand(input)), mutate)
    x = x + _drop(3, @contract(hidden), mutate)
    if @norm_first then x else @norms[2](x)

  def forward(x, memory, mask: null, memory_mask: null, causal: true):
    _forward(x, memory, mask, memory_mask, causal, false)

  def forward!(x, memory, mask: null, memory_mask: null, causal: true):
    _forward(x, memory, mask, memory_mask, causal, true)
