package sputniktorch.nn.transformer_encoder_layer

from sputniktorch.nn.module import Module
from sputniktorch.nn.multihead_attention import MultiheadAttention
from sputniktorch.nn.linear import Linear
from sputniktorch.nn.layer_norm import LayerNorm
from sputniktorch.nn.gelu import GELU
from sputniktorch.nn.dropout import Dropout

export TransformerEncoderLayer

class TransformerEncoderLayer < Module:
  def init(d_model, nhead, dim_feedforward: 2048, dropout: 0.1,
      @norm_first: false, seed: 0, device: "cpu"):
    _initialize_registry!
    @attention = MultiheadAttention(d_model, nhead, seed:, device:)
    @expand = Linear(d_model, dim_feedforward, seed: seed + 4, device:)
    @contract = Linear(dim_feedforward, d_model, seed: seed + 5, device:)
    @activation = GELU()
    @norm1 = LayerNorm(d_model, device:)
    @norm2 = LayerNorm(d_model, device:)
    @drop_attention = Dropout(p: dropout, seed: seed + 6)
    @drop_hidden = Dropout(p: dropout, seed: seed + 7)
    @drop_output = Dropout(p: dropout, seed: seed + 8)

  def _drop(layer, x, mutate):
    if mutate then layer.call!(x) else layer(x)

  def _feedforward(x, mutate):
    hidden = _drop(@drop_hidden, @activation(@expand(x)), mutate)
    _drop(@drop_output, @contract(hidden), mutate)

  def _forward(x, mask, causal, mutate):
    attended = @attention(if @norm_first then @norm1(x) else x, mask:, causal:)
    residual = x + _drop(@drop_attention, attended, mutate)
    if @norm_first:
      residual + _feedforward(@norm2(residual), mutate)
    else:
      residual = @norm1(residual)
      @norm2(residual + _feedforward(residual, mutate))

  def forward(x, mask: null, causal: false): _forward(x, mask, causal, false)
  def forward!(x, mask: null, causal: false): _forward(x, mask, causal, true)
