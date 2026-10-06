package sputniktorch.nn.transformer_decoder

from sputniktorch.nn.module import Module
from sputniktorch.nn.module_list import ModuleList
from sputniktorch.nn.transformer_decoder_layer import TransformerDecoderLayer

export TransformerDecoder

class TransformerDecoder < Module:
  attr layers

  def init(d_model, nhead, num_layers: 6, dim_feedforward: 2048,
      dropout: 0.1, norm_first: false, seed: 0, device: "cpu"):
    _initialize_registry!
    raise ValueError("num_layers must be positive") if num_layers <= 0
    @layers = ModuleList(Array.of(num_layers) |index|:
      TransformerDecoderLayer(d_model, nhead, dim_feedforward:,
        dropout:, norm_first:, seed: seed + index * 14, device:)
    )

  def forward(x, memory, mask: null, memory_mask: null, causal: true):
    @layers.each |layer|:
      x = layer(x, memory, mask:, memory_mask:, causal:)
    x

  def forward!(x, memory, mask: null, memory_mask: null, causal: true):
    @layers.each |layer|:
      x = layer.call!(x, memory, mask:, memory_mask:, causal:)
    x
