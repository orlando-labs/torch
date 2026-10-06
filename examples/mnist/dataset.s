package sputniktorch.mnist.dataset

from sputniktorch.data import TensorDataset
from sputniktorch.mnist.idx import read_idx

export MNIST

class MNIST < TensorDataset:
  def init(directory, split:, limit: null):
    raise ValueError("MNIST split must be train or test") unless split in [:train, :test]
    prefix = if (split == :train) then "train" else "t10k"
    images = read_idx("#{directory}/#{prefix}-images-idx3-ubyte")
    labels = read_idx("#{directory}/#{prefix}-labels-idx1-ubyte")
    if images.shape != [labels.numel, 28, 28] or labels.rank != 1:
      raise ValueError("MNIST image and label dimensions disagree")
    if limit != null:
      raise ValueError("MNIST limit must be within the split") unless limit in 1..labels.numel
      images = images.narrow(0, 0, limit)
      labels = labels.narrow(0, 0, limit)
    # Keep the corpus on CPU; transfer each batch to the training device.
    features = (images.unsqueeze(1).to(device: "cpu", dtype: "float32") / 255.0 - 0.1307) / 0.3081
    _initialize_tensor_dataset!((features, labels.to(device: "cpu", dtype: "int64")))
