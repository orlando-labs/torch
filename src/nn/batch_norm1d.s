package sputniktorch.nn.batch_norm1d

from sputniktorch.nn.batch_norm import BatchNorm

export BatchNorm1d

class BatchNorm1d < BatchNorm:
  def init(features, eps: 1e-5, momentum: 0.1, affine: true,
      track_running_stats: true, device: "cpu"):
    _initialize_batch_norm!(1, features, eps:, momentum:,
      affine:, track_running_stats:, device:)

