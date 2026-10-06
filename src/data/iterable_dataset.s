package sputniktorch.data.iterable_dataset

export IterableDataset

class IterableDataset:
  def each(&block): raise NoMethodError("IterableDataset must implement each")
