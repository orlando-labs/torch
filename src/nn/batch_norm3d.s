package torch.nn.batch_norm3d

from torch.nn.batch_norm import BatchNorm

export BatchNorm3d

class BatchNorm3d < BatchNorm:
  def init(features, eps: 1e-5, momentum: 0.1, affine: true,
      track_running_stats: true, device: "cpu", dtype: "float32", bias: true):
    _initialize_batch_norm!(3, features, eps:, momentum:,
      affine:, track_running_stats:, device:, dtype:, bias:)
