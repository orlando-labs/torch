package torch.nn.cosine_similarity

from torch.nn.module import Module
from torch.nn.functional import cosine_similarity

export CosineSimilarity

class CosineSimilarity < Module:
  def init(@dim: 1, @eps: 1e-8): _initialize_registry!
  def forward(input1, input2): cosine_similarity(input1, input2, dim: @dim, eps: @eps)
