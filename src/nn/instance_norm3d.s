package torch.nn.instance_norm3d

from torch.nn.instance_norm import InstanceNorm

export InstanceNorm3d

class InstanceNorm3d < InstanceNorm:
  def init(features, eps: 1e-5, momentum: 0.1, affine: false,
      track_running_stats: false, device: "cpu", dtype: "float32", bias: true):
    _initialize_batch_norm!(3, features, eps:, momentum:, affine:,
      track_running_stats:, device:, dtype:, bias:)
