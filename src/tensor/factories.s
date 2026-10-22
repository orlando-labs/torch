package torch.tensor.factories

from torch.tensor.core import make, factory
from torch.tensor.value import Tensor
from torch.device import device_text

export tensor, zeros, ones, full, empty, eye, arange, linspace, randn, rand
export zeros_like, ones_like, full_like, empty_like

def tensor(data, dtype: "auto", device: "cpu", requires_grad: false):
  Tensor(make(data, dtype.to_str, device_text(device), requires_grad))

def zeros(shape, dtype: "float32", device: "cpu", requires_grad: false):
  Tensor(factory("zeros", shape.array, dtype.to_str, device_text(device), requires_grad, 0))

def ones(shape, dtype: "float32", device: "cpu", requires_grad: false):
  Tensor(factory("ones", shape.array, dtype.to_str, device_text(device), requires_grad, 0))

def randn(shape, seed:, dtype: "float32", device: "cpu", requires_grad: false):
  Tensor(factory("randn", shape.array, dtype.to_str, device_text(device), requires_grad, seed))

def rand(shape, seed:, dtype: "float32", device: "cpu", requires_grad: false):
  Tensor(factory("rand", shape.array, dtype.to_str, device_text(device), requires_grad, seed))

def full(shape, value, dtype: "float32", device: "cpu", requires_grad: false):
  Tensor(factory("full", shape.array, dtype.to_str, device_text(device), requires_grad, value))

def empty(shape, dtype: "float32", device: "cpu", requires_grad: false):
  Tensor(factory("empty", shape.array, dtype.to_str, device_text(device), requires_grad, 0))

def eye(rows, columns: null, dtype: "float32", device: "cpu", requires_grad: false):
  shape = [rows, (if columns == null then rows else columns)]
  Tensor(factory("eye", shape, dtype.to_str, device_text(device), requires_grad, 0))

def arange(stop, start: 0, step: 1, dtype: "int64", device: "cpu"):
  Tensor(factory("arange", [], dtype.to_str, device_text(device), false, [start, stop, step]))

def linspace(start, stop, steps:, dtype: "float32", device: "cpu"):
  Tensor(factory("linspace", [], dtype.to_str, device_text(device), false, [start, stop, steps]))

def zeros_like(x, requires_grad: false): zeros(x.shape, dtype: x.dtype, device: x.device, requires_grad:)
def ones_like(x, requires_grad: false): ones(x.shape, dtype: x.dtype, device: x.device, requires_grad:)
def full_like(x, value, requires_grad: false): full(x.shape, value, dtype: x.dtype, device: x.device, requires_grad:)
def empty_like(x, requires_grad: false): empty(x.shape, dtype: x.dtype, device: x.device, requires_grad:)
