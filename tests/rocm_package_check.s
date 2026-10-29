from torch.tensor.factories import tensor
from torch.backends import version, rocm_available?, synchronize
from torch.optim.sgd import SGD

raise ValueError("ROCm unavailable") unless rocm_available?
x = tensor([1.0, 2.0, 3.0], device: "cuda:0", requires_grad: true)
loss = x.square.sum
loss.backward!
raise ValueError("AMD forward/backward") unless x.device.type == "cuda" and loss.item == 14.0 and x.grad.to_a == [2.0, 4.0, 6.0]
matrix = tensor([[1.0, 2.0], [3.0, 4.0]], device: "cuda:0", requires_grad: true)
product = matrix.matmul(tensor([[1.0], [2.0]], device: "cuda:0"))
raise ValueError("AMD matrix product") unless product.to_a == [[5.0], [11.0]]
product.sum.backward!
raise ValueError("AMD matrix gradient") unless matrix.grad.to_a == [[1.0, 2.0], [1.0, 2.0]]
weight = tensor([0.0], device: "cuda:0", requires_grad: true)
optimizer = SGD([weight], lr: 0.05)
60.times:
  optimizer.zero_grad!
  error = (weight * 2 - 6).square.mean
  error.backward!
  optimizer.step!
raise ValueError("AMD training update") unless (weight.item - 3.0).abs < 0.0001
synchronize
message = "PASS ROCm LibTorch #{version} forward=14 gradient=[2,4,6] matmul=OK trained_weight=#{weight.item}"
print message
message
