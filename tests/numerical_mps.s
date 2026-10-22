package torch.numerical_mps_test

import torch.fft as fft
import torch.linalg as la
import torch.special as special
from torch import tensor, complex
from torch.checks import check

export main

def close(x, y):
  check(x.device.type == "mps" and y.device.type == "mps", "actual MPS placement")
  check(x.shape == y.shape and (x - y).abs.sum.item < 1e-4, "MPS numerical result")

def main():
  x = tensor([1.0, 2.0, 3.0, 4.0], device: "mps", requires_grad: true)
  norm = la.vector_norm(x)
  check(norm.device.type == "mps", "MPS norm")
  norm.backward!
  close(x.grad, x.detach / norm.detach)
  close(special.erf(x.detach) + special.erfc(x.detach), x.detach * 0 + 1)
  z = complex(x.detach, x.detach * 2)
  close(z.real, x.detach)
  close(z.imag, x.detach * 2)
  close(fft.ifft(fft.fft(z)).real, x.detach)
  close(fft.irfft(fft.rfft(x.detach), n: 4), x.detach)
  print "PASS torch numerical MPS"
  0
