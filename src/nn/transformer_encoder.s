package sputniktorch.nn.transformer_encoder

from sputniktorch.nn.module import Module
from sputniktorch.nn.module_list import ModuleList
from sputniktorch.nn.transformer_encoder_layer import TransformerEncoderLayer

export TransformerEncoder

# Every layer owns independent parameters; seeds are explicit and stable.
class TransformerEncoder < Module:
  attr layers

  def init(d_model, nhead, num_layers: 6, dim_feedforward: 2048,
      dropout: 0.1, norm_first: false, seed: 0, device: "cpu"):
    _initialize_registry!
    raise ValueError("num_layers must be positive") if num_layers <= 0
    @layers = ModuleList(Array.of(num_layers) |index|:
      TransformerEncoderLayer(d_model, nhead, dim_feedforward:,
        dropout:, norm_first:, seed: seed + index * 9, device:)
    )

  def forward(x, mask: null, causal: false):
    @layers.each |layer|:
      x = layer(x, mask:, causal:)
    x

  def forward!(x, mask: null, causal: false):
    @layers.each |layer|:
      x = layer.call!(x, mask:, causal:)
    x
