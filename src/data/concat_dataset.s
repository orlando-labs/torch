package sputniktorch.data.concat_dataset

from sputniktorch.data.dataset import Dataset

export ConcatDataset

class ConcatDataset < Dataset:
  def init(datasets): @datasets = datasets.array
  def count(): @datasets.reduce(0) |total, dataset|: total + dataset.count

  def [](index):
    offset = if index < 0 then count + index else index
    raise IndexError("dataset index outside bounds") if offset < 0 or offset >= count
    @datasets.each |dataset|:
      return dataset[offset] if offset < dataset.count
      offset -= dataset.count
