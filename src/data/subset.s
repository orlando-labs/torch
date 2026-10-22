package torch.data.subset

from torch.data.dataset import Dataset

export Subset

class Subset < Dataset:
  def init(@dataset, indices): @indices = indices.array
  def count(): @indices.count
  def [](index): @dataset[@indices[index]]
