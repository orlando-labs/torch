package torch.nn.embedding

from torch.nn.module import Module
from torch.nn.embedding_functional import embedding, embedding!, padding_index, embedding_weight
from torch.tensor.factories import randn
from torch.autograd import no_grad

export Embedding

class Embedding < Module:
  attr weight
  attr num_embeddings
  attr embedding_dim
  attr padding_idx

  def init(num_embeddings, embedding_dim, padding_idx: null, max_norm: null, norm_type: 2.0,
      scale_grad_by_freq: false, sparse: false, weight: null, freeze: false,
      seed: 0, device: "cpu", dtype: "float32"):
    _initialize_embedding!(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
      scale_grad_by_freq, sparse, weight, freeze, seed, device, dtype)

  def _initialize_embedding!(vocabulary, dimensions, padding, max_norm, norm_type,
      frequency, sparse, initial, freeze, seed, device, dtype):
    _initialize_registry!
    @num_embeddings = vocabulary
    @embedding_dim = dimensions
    @padding_idx = padding_index(padding, vocabulary)
    @max_norm = max_norm
    @norm_type = norm_type
    @scale_grad_by_freq = frequency
    @sparse = sparse
    @weight = embedding_weight(vocabulary, dimensions, @padding_idx, initial, freeze, seed, device, dtype)
    register_parameter!("weight", @weight)

  def stateful?(): @max_norm != null

  def forward(indices):
    raise ValueError("max_norm changes embedding weight; use call! or forward!") if stateful?
    embedding(indices, @weight, padding_idx: @padding_idx,
      scale_grad_by_freq: @scale_grad_by_freq, sparse: @sparse)

  def forward!(indices):
    embedding!(indices, @weight, padding_idx: @padding_idx, max_norm: @max_norm,
      norm_type: @norm_type, scale_grad_by_freq: @scale_grad_by_freq, sparse: @sparse)

  def reset_parameters!(seed: 0):
    no_grad():
      @weight.copy_from!(randn(@weight.shape, seed:, device: @weight.device, dtype: @weight.dtype))
      @weight[@padding_idx].zero! if @padding_idx != null
    self

  class_method def from_pretrained(weights, freeze: true, padding_idx: null, max_norm: null,
      norm_type: 2.0, scale_grad_by_freq: false, sparse: false):
    raise ValueError("pretrained weight must be a 2D Tensor") unless weights.rank == 2
    Embedding(weights.shape[0], weights.shape[1], weight: weights, freeze:, padding_idx:,
      max_norm:, norm_type:, scale_grad_by_freq:, sparse:)
