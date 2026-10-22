package torch.nn.transformer_encoder

from torch.nn.module import Module
from torch.nn.module_list import ModuleList
from torch.nn.transformer_encoder_layer import TransformerEncoderLayer

export TransformerEncoder

# Every layer owns independent parameters; seeds are explicit and stable.
class TransformerEncoder < Module:
  attr layers

  def init(d_model, nhead, num_layers: 6, dim_feedforward: 2048,
      dropout: 0.1, norm_first: false, batch_first: true, bias: true,
      layer_norm_eps: 1e-5, activation: :gelu, seed: 0, device: "cpu", dtype: "float32"):
    _initialize_registry!
    raise ValueError("num_layers must be positive") if num_layers <= 0
    @layers = ModuleList(Array.of(num_layers) |index|:
      TransformerEncoderLayer(d_model, nhead, dim_feedforward:,
        dropout:, norm_first:, batch_first:, bias:, layer_norm_eps:, activation:,
        seed: seed + index * 9, device:, dtype:)
    )

  def forward(x, mask: null, attn_mask: null, key_padding_mask: null, causal: false):
    @layers.each |layer|:
      x = layer(x, mask:, attn_mask:, key_padding_mask:, causal:)
    x

  def forward!(x, mask: null, attn_mask: null, key_padding_mask: null, causal: false):
    @layers.each |layer|:
      x = layer.call!(x, mask:, attn_mask:, key_padding_mask:, causal:)
    x
