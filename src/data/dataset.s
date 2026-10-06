package sputniktorch.data.dataset

export Dataset

class Dataset:
  def [](index): raise NoMethodError("Dataset must implement [](index)")
  def count(): raise NoMethodError("Dataset must implement count")
