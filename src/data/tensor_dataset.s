package sputniktorch.data.tensor_dataset

from sputniktorch.data.dataset import Dataset
from sputniktorch.tensor.factories import tensor
from sputniktorch.tensor.value import Tensor
from sputniktorch.tensor.foreach import index_select

export TensorDataset

class TensorDataset < Dataset:
  def init(*tensors): _initialize_tensor_dataset!(tensors)

  def _initialize_tensor_dataset!(tensors):
    raise ValueError("TensorDataset needs at least one tensor") if tensors.empty?
    @tensors = tensors
    @cores = tensors.map: $it._core
    @cpu_tensors = tensors.all?: $it.device.to_str == "cpu"
    @count = tensors[0].shape[0]
    matches = tensors.all?: $it.shape[0] == @count
    raise ValueError("dataset tensors must have equal leading dimensions") unless matches

  attr count from @count
  def [](index): (@tensors.map: $it[index]).tuple

  def batch(indices):
    order = tensor(indices, dtype: "int64")
    if @cpu_tensors:
      (index_select(@cores, 0, order._core).map: Tensor($it)).tuple
    else:
      (@tensors.map: $it.index_select(0, order.to(device: $it.device))).tuple
