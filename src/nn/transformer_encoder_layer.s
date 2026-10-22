package torch.nn.transformer_encoder_layer

from torch.nn.module import Module
from torch.nn.multihead_attention import MultiheadAttention
from torch.nn.linear import Linear
from torch.nn.layer_norm import LayerNorm
from torch.nn.gelu import GELU
from torch.nn.relu import ReLU
from torch.nn.dropout import Dropout

export TransformerEncoderLayer

class TransformerEncoderLayer < Module:
  def init(d_model, nhead, dim_feedforward: 2048, dropout: 0.1,
      @norm_first: false, batch_first: true, bias: true, layer_norm_eps: 1e-5,
      activation: :gelu, seed: 0, device: "cpu", dtype: "float32"):
    _initialize_registry!
    raise ValueError("activation must be :gelu or :relu") unless activation.str in ["gelu", "relu"]
    @attention = MultiheadAttention(d_model, nhead, dropout:, batch_first:, bias:, seed:, device:, dtype:)
    @expand = Linear(d_model, dim_feedforward, bias:, seed: seed + 4, device:, dtype:)
    @contract = Linear(dim_feedforward, d_model, bias:, seed: seed + 5, device:, dtype:)
    @activation = if activation.str == "gelu" then GELU() else ReLU()
    @norm1 = LayerNorm(d_model, eps: layer_norm_eps, bias:, device:, dtype:)
    @norm2 = LayerNorm(d_model, eps: layer_norm_eps, bias:, device:, dtype:)
    @drop_attention = Dropout(p: dropout, seed: seed + 6)
    @drop_hidden = Dropout(p: dropout, seed: seed + 7)
    @drop_output = Dropout(p: dropout, seed: seed + 8)

  def _drop(layer, x, mutate):
    if mutate then layer.call!(x) else layer(x)

  def _feedforward(x, mutate):
    hidden = _drop(@drop_hidden, @activation(@expand(x)), mutate)
    _drop(@drop_output, @contract(hidden), mutate)

  def _forward(x, mask, attn_mask, key_padding_mask, causal, mutate):
    input = if @norm_first then @norm1(x) else x
    attended = if mutate:
      @attention.call!(input, mask:, attn_mask:, key_padding_mask:, causal:)
    else:
      @attention(input, mask:, attn_mask:, key_padding_mask:, causal:)
    residual = x + _drop(@drop_attention, attended, mutate)
    if @norm_first:
      residual + _feedforward(@norm2(residual), mutate)
    else:
      residual = @norm1(residual)
      @norm2(residual + _feedforward(residual, mutate))

  def forward(x, mask: null, attn_mask: null, key_padding_mask: null, causal: false):
    _forward(x, mask, attn_mask, key_padding_mask, causal, false)

  def forward!(x, mask: null, attn_mask: null, key_padding_mask: null, causal: false):
    _forward(x, mask, attn_mask, key_padding_mask, causal, true)
