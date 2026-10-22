package torch.nn.transformer

from torch.nn.module import Module
from torch.nn.transformer_encoder import TransformerEncoder
from torch.nn.transformer_decoder import TransformerDecoder

export Transformer

class Transformer < Module:
  attr encoder
  attr decoder

  def init(d_model, nhead, num_encoder_layers: 6, num_decoder_layers: 6,
      dim_feedforward: 2048, dropout: 0.1, norm_first: false, batch_first: true,
      bias: true, layer_norm_eps: 1e-5, activation: :gelu, seed: 0, device: "cpu", dtype: "float32"):
    _initialize_registry!
    @encoder = TransformerEncoder(d_model, nhead, num_layers: num_encoder_layers,
      dim_feedforward:, dropout:, norm_first:, batch_first:, bias:, layer_norm_eps:, activation:,
      seed:, device:, dtype:)
    @decoder = TransformerDecoder(d_model, nhead, num_layers: num_decoder_layers,
      dim_feedforward:, dropout:, norm_first:, batch_first:, bias:, layer_norm_eps:, activation:,
      seed: seed + num_encoder_layers * 9, device:, dtype:)

  def forward(source, target, source_mask: null, target_mask: null, memory_mask: null,
      source_attn_mask: null, target_attn_mask: null, memory_attn_mask: null,
      source_key_padding_mask: null, target_key_padding_mask: null, memory_key_padding_mask: null,
      source_causal: false, target_causal: true, memory_causal: false):
    memory = @encoder(source, mask: source_mask, attn_mask: source_attn_mask,
      key_padding_mask: source_key_padding_mask, causal: source_causal)
    @decoder(target, memory, mask: target_mask, memory_mask:, attn_mask: target_attn_mask,
      memory_attn_mask:, key_padding_mask: target_key_padding_mask, memory_key_padding_mask:,
      causal: target_causal, memory_causal:)

  def forward!(source, target, source_mask: null, target_mask: null, memory_mask: null,
      source_attn_mask: null, target_attn_mask: null, memory_attn_mask: null,
      source_key_padding_mask: null, target_key_padding_mask: null, memory_key_padding_mask: null,
      source_causal: false, target_causal: true, memory_causal: false):
    memory = @encoder.call!(source, mask: source_mask, attn_mask: source_attn_mask,
      key_padding_mask: source_key_padding_mask, causal: source_causal)
    @decoder.call!(target, memory, mask: target_mask, memory_mask:, attn_mask: target_attn_mask,
      memory_attn_mask:, key_padding_mask: target_key_padding_mask, memory_key_padding_mask:,
      causal: target_causal, memory_causal:)
