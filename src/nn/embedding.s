package sputniktorch.nn.embedding

from sputniktorch.nn.module import Module
from sputniktorch.nn.functional import embedding
from sputniktorch.tensor.factories import randn
from sputniktorch.autograd import no_grad

export Embedding

class Embedding < Module:
  attr weight

  def init(num_embeddings, embedding_dim, @padding_idx: -1, seed: 0, device: "cpu"):
    _initialize_registry!
    if num_embeddings <= 0 or embedding_dim <= 0:
      raise ValueError("embedding dimensions must be positive")
    if padding_idx < -1 or padding_idx >= num_embeddings:
      raise ValueError("padding_idx is outside the vocabulary")
    @weight = randn([num_embeddings, embedding_dim], seed:, device:, requires_grad: true)
    if padding_idx >= 0:
      no_grad(): @weight[padding_idx].zero!
    register_parameter!("weight", @weight)

  def forward(indices): embedding(indices, @weight, padding_idx: @padding_idx)
