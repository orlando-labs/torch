from slowpilot.torch.tensor.factories import tensor
from slowpilot.torch.backends import version, mps_available?, synchronize
from slowpilot.torch.tensor.bytes import from_bytes
from slowpilot.torch.random.generator import Generator
from slowpilot.torch.errors import DeviceError

raise ValueError("Expected LibTorch 2.14.0") unless version() == "2.14.0"
raise ValueError("MPS is unavailable") unless mps_available?()
["cpu", "mps"].each |device|:
  x = tensor([1.0, 2.0, 3.0], device:, requires_grad: true)
  raise ValueError("Incorrect tensor device") unless x.device.type == device
  loss = x.square.sum
  loss.backward!
  raise ValueError("Incorrect forward") unless loss.item == 14.0
  raise ValueError("Incorrect gradient") unless x.grad.to_a == [2.0, 4.0, 6.0]
raise ValueError("CPU-to-MPS transfer") unless tensor([4.0]).to(device: "mps").to_a == [4.0]
rejected = false
try:
  tensor([1.0], device: "mps:63")
rescue DeviceError:
  rejected = true
raise ValueError("Scoped MPS device error") unless rejected
raw = from_bytes("abc".bytes, device: "mps")
raise ValueError("MPS bytes") unless raw.device.type == "mps" and raw.to_a == [97, 98, 99]
cpu_random = Generator(seed: 42).randn!([4])
mps_random = Generator(seed: 42).randn!([4], device: "mps")
raise ValueError("MPS generation") unless mps_random.device.type == "mps" and mps_random.to_a == cpu_random.to_a
synchronize()
print "PASS CPU/MPS LibTorch #{version()} forward=14 gradient=[2,4,6] bytes=OK generator=OK scoped-error=OK"
