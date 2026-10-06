package sputniktorch.nn.batch_norm2d

from sputniktorch.nn.batch_norm import BatchNorm

export BatchNorm2d

class BatchNorm2d < BatchNorm:
  def init(features, eps: 1e-5, momentum: 0.1, affine: true,
      track_running_stats: true, device: "cpu"):
    _initialize_batch_norm!(2, features, eps:, momentum:,
      affine:, track_running_stats:, device:)

