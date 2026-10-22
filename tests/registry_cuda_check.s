from slowpilot.torch.tensor.factories import tensor
from slowpilot.torch.backends import version, cuda_available?, cuda_device_count, synchronize
from slowpilot.torch.tensor.bytes import from_bytes
from slowpilot.torch.random.generator import Generator
from slowpilot.torch.errors import DeviceError

raise ValueError("Expected LibTorch 2.14.0") unless version().starts_with?("2.14.0")
raise ValueError("CUDA backend is unavailable") unless cuda_available?() and cuda_device_count() > 0
x = tensor([1.0, 2.0, 3.0], device: "cuda:0", requires_grad: true)
raise ValueError("Tensor was not placed on CUDA") unless x.device.type == "cuda"
loss = x.square.sum
loss.backward!
raise ValueError("Incorrect CUDA forward result") unless loss.item == 14.0
raise ValueError("Incorrect CUDA gradient") unless x.grad.to_a == [2.0, 4.0, 6.0]
raise ValueError("CPU to CUDA transfer") unless tensor([4.0]).to(device: "cuda:0").to_a == [4.0]
rejected = false
try:
  tensor([1.0], device: "cuda:63")
rescue DeviceError:
  rejected = true
raise ValueError("CUDA device error did not retain its scoped class") unless rejected
raw = from_bytes("abc".bytes, device: "cuda:0")
raise ValueError("CUDA from_bytes placement or value") unless raw.device.type == "cuda" and raw.to_a == [97, 98, 99]
cpu_random = Generator(seed: 42).randn!([4])
cuda_random = Generator(seed: 42).randn!([4], device: "cuda:0")
raise ValueError("CUDA generator placement or reproducibility") unless cuda_random.device.type == "cuda" and cuda_random.to_a == cpu_random.to_a
synchronize()
message = "PASS CUDA LibTorch #{version()} forward=14 gradient=[2,4,6] bytes=OK generator=OK device=#{x.device}"
print message
message
