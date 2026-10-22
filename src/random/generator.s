package torch.random.generator

from torch.random.core import make as _make, sample! as _sample!
from torch.tensor.value import Tensor
from torch.device import device_text

export Generator

class Generator:
  def init(seed: 0): @core = _make(seed)
  def state(): Tensor(_sample!(@core, "state", null, "float32", "cpu"))

  def load_state!(state):
    _sample!(@core, "restore", state._core, "float32", "cpu")
    self

  def seed!(seed):
    _sample!(@core, "seed", seed, "float32", "cpu")
    self

  def rand!(shape, dtype: "float32", device: "cpu"):
    Tensor(_sample!(@core, "rand", shape.array, dtype.to_str, device_text(device)))

  def randn!(shape, dtype: "float32", device: "cpu"):
    Tensor(_sample!(@core, "randn", shape.array, dtype.to_str, device_text(device)))

  def randperm!(count, device: "cpu"):
    Tensor(_sample!(@core, "randperm", [count], "float32", device_text(device)))
