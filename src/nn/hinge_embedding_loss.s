package torch.nn.hinge_embedding_loss

from torch.nn.module import Module
from torch.nn.functional import hinge_embedding_loss

export HingeEmbeddingLoss

class HingeEmbeddingLoss < Module:
  def init(@margin: 1.0, @reduction: :mean): _initialize_registry!
  def forward(input, target): hinge_embedding_loss(input, target, margin: @margin, reduction: @reduction)
