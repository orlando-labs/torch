package torch.nn.embedding_bag

from torch.nn.embedding import Embedding
from torch.nn.embedding_functional import embedding_bag, embedding_bag!, bag_mode

export EmbeddingBag

class EmbeddingBag < Embedding:
  def init(num_embeddings, embedding_dim, mode: :mean, padding_idx: null,
      max_norm: null, norm_type: 2.0, scale_grad_by_freq: false, sparse: false,
      @include_last_offset: false, weight: null, freeze: false,
      seed: 0, device: "cpu", dtype: "float32"):
    bag_mode(mode, scale_grad_by_freq, sparse)
    @mode = mode
    _initialize_embedding!(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
      scale_grad_by_freq, sparse, weight, freeze, seed, device, dtype)

  def forward(input, offsets = null, per_sample_weights: null):
    raise ValueError("max_norm changes embedding weight; use call! or forward!") if stateful?
    embedding_bag(input, @weight, offsets, mode: @mode, padding_idx: @padding_idx,
      scale_grad_by_freq: @scale_grad_by_freq, sparse: @sparse, per_sample_weights:,
      include_last_offset: @include_last_offset)

  def forward!(input, offsets = null, per_sample_weights: null):
    embedding_bag!(input, @weight, offsets, mode: @mode, padding_idx: @padding_idx,
      max_norm: @max_norm, norm_type: @norm_type, scale_grad_by_freq: @scale_grad_by_freq,
      sparse: @sparse, per_sample_weights:, include_last_offset: @include_last_offset)

  class_method def from_pretrained(weights, freeze: true, mode: :mean, padding_idx: null,
      max_norm: null, norm_type: 2.0, scale_grad_by_freq: false, sparse: false, include_last_offset: false):
    raise ValueError("pretrained weight must be a 2D Tensor") unless weights.rank == 2
    EmbeddingBag(weights.shape[0], weights.shape[1], weight: weights, freeze:, mode:, padding_idx:,
      max_norm:, norm_type:, scale_grad_by_freq:, sparse:, include_last_offset:)
