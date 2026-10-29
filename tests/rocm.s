package torch.rocm_test

from torch.tensor.bytes import from_bytes
from torch.random.generator import Generator
from torch.backends import version, rocm_built?, rocm_available?, cuda_device_count, synchronize
from torch.checks import run

export main

def main():
  raise ValueError("Expected ROCm LibTorch") unless rocm_built? and rocm_available?
  raise ValueError("No AMD devices") unless cuda_device_count > 0
  run("cuda")
  raw = from_bytes("abc".bytes, device: "cuda:0")
  raise ValueError("ROCm bytes transfer") unless raw.device.type == "cuda" and raw.to_a == [97, 98, 99]
  cpu = Generator(seed: 42).randn!([4])
  gpu = Generator(seed: 42).randn!([4], device: "cuda:0")
  raise ValueError("ROCm generator") unless gpu.device.type == "cuda" and gpu.to_a == cpu.to_a
  synchronize
  print "PASS ROCm LibTorch #{version} devices=#{cuda_device_count} bytes=OK generator=OK"
