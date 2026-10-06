package sputniktorch.mnist.idx

from sputniktorch import from_bytes

export read_idx

def _uint32(bytes, offset):
  (0...4).reduce(0) |value, index|: value * 256 + bytes[offset + index]

# MNIST stores uint8 tensors with big-endian dimensions in an IDX header.
def read_idx(path):
  bytes = fs.read_bytes(path, limit: 64 * 1024 * 1024)
  raise ValueError("truncated IDX header: #{path}") if bytes.count < 4
  if bytes[0] != 0 or bytes[1] != 0 or bytes[2] != 8:
    raise ValueError("expected uint8 IDX data: #{path}")
  rank = bytes[3]
  raise ValueError("invalid IDX rank: #{path}") unless rank in 1..8
  offset = 4 + rank * 4
  raise ValueError("truncated IDX dimensions: #{path}") if bytes.count < offset
  shape = Array.of(rank): _uint32(bytes, 4 + $it * 4)
  raise ValueError("empty IDX dimension: #{path}") if shape.any?: $it == 0
  from_bytes(bytes, offset:, shape:)
