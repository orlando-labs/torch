package sputniktorch.nn.transformer

from sputniktorch.nn.module import Module
from sputniktorch.nn.transformer_encoder import TransformerEncoder
from sputniktorch.nn.transformer_decoder import TransformerDecoder

export Transformer

class Transformer < Module:
  attr encoder
  attr decoder

  def init(d_model, nhead, num_encoder_layers: 6, num_decoder_layers: 6,
      dim_feedforward: 2048, dropout: 0.1, norm_first: false, seed: 0, device: "cpu"):
    _initialize_registry!
    @encoder = TransformerEncoder(d_model, nhead, num_layers: num_encoder_layers,
      dim_feedforward:, dropout:, norm_first:,
      seed:, device:)
    @decoder = TransformerDecoder(d_model, nhead, num_layers: num_decoder_layers,
      dim_feedforward:, dropout:, norm_first:,
      seed: seed + num_encoder_layers * 9, device:)

  def forward(source, target, source_mask: null, target_mask: null, memory_mask: null):
    memory = @encoder(source, mask: source_mask)
    @decoder(target, memory, mask: target_mask, memory_mask:)

  def forward!(source, target, source_mask: null, target_mask: null, memory_mask: null):
    memory = @encoder.call!(source, mask: source_mask)
    @decoder.call!(target, memory, mask: target_mask, memory_mask:)
