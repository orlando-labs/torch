package sputniktorch.distributed_errors_test

import sputniktorch
from sputniktorch.checks import check

export main

def main():
  group = sputniktorch.group_from_env(timeout_ms: 2000)
  try:
    model = sputniktorch.Linear(1, group.rank + 1, seed: 7)
    ddp = sputniktorch.DDP(model, group)
    mismatch = false
    try:
      ddp.synchronize!
    rescue Sputniktorch.Errors.DistributedError |e|:
      mismatch = true
    check(mismatch, "DDP schema mismatch must be detected on every rank")
    group.barrier!
    print("PASS DDP schema mismatch rank " + group.rank.to_str)
  ensure:
    group.close!
  group.close!
  closed = false
  try:
    group.rank
  rescue LifetimeError |e|:
    closed = true
  check(closed, "process group use after close")
  0
