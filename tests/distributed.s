package sputniktorch.distributed_test

import sputniktorch
from sputniktorch.checks import check

export main

def main():
  group = sputniktorch.group_from_env()
  rank = group.rank
  try:
    value = sputniktorch.tensor([rank + 1.0])
    group.all_reduce!(value)
    check(value.item == 3.0, "all-reduce SUM")
    value.fill!(if rank == 0 then 42 else 0)
    group.broadcast!(value)
    check(value.item == 42.0, "broadcast")
    model = sputniktorch.Linear(1, 1, seed: rank)
    ddp = sputniktorch.DDP(model, group)
    ddp.synchronize!
    reference = sputniktorch.Linear(1, 1, seed: 99)
    reference.load_state_dict!(model.state_dict)
    all_x = sputniktorch.tensor([[1.0], [2.0]])
    reference.call(all_x).sub(all_x.mul(2).add(1)).square.mean().backward!()
    x = sputniktorch.tensor([[rank + 1.0]])
    ddp.backward!(ddp.call(x).sub(x.mul(2).add(1)).square.mean())
    expected = reference.named_parameters
    model.named_parameters.each |name, parameter|:
      check(parameter.grad.sub(expected[name].grad).square.sum().item < 0.000001, "DDP gradient matches concatenated batch")
    optimizer = sputniktorch.SGD(ddp.parameters, lr: 0.05)
    60.times:
      optimizer.zero_grad!
      ddp.backward!(ddp.call(x).sub(x.mul(2).add(1)).square.mean())
      optimizer.step!
    check(ddp.call(x).sub(x.mul(2).add(1)).square.mean().item < 0.1, "DDP training")
    group.barrier!
    print("PASS sputniktorch Gloo + DDP rank " + rank.to_str)
  ensure:
    group.close!
  0
