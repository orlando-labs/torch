package torch.nn.cosine_embedding_loss

from torch.nn.module import Module
from torch.nn.functional import cosine_embedding_loss

export CosineEmbeddingLoss

class CosineEmbeddingLoss < Module:
  def init(@margin: 0.0, @reduction: :mean): _initialize_registry!

  def forward(input1, input2, target):
    cosine_embedding_loss(input1, input2, target, margin: @margin, reduction: @reduction)
