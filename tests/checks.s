package sputniktorch.checks

import sputniktorch
export run, check

def check(condition, message):
  if not condition:
    raise ValueError(message)

def run(device):
  x = sputniktorch.tensor([[1.0, 2.0], [3.0, 4.0]], device:, requires_grad: true)
  check(x.shape == [2, 2], "shape")
  check(x.device.type == device, "actual device")
  loss = x.square.sum()
  loss.backward!()
  check(loss.item == 30.0, "forward")
  check(x.grad.to_a == [[2.0, 4.0], [6.0, 8.0]], "backward")
  x.zero_grad!
  check(x.grad == null, "zero_grad set_to_none")

  released = sputniktorch.tensor([1.0], device:)._core
  check(released.destroy!, "bare collected handle teardown")
  check(not released.destroy!, "collected handle teardown is idempotent")
  closed = false
  try:
    sputniktorch.Tensor(released).shape
  rescue LifetimeError |error|:
    closed = true
  check(closed, "collected handle access after teardown")

  octets = "ABC".bytes
  imported = sputniktorch.from_bytes(octets, offset: 1, shape: [1, 2], device:)
  check(imported.to_a == [[66, 67]] and imported.dtype.to_str == "uint8", "Bytes tensor import")
  imported.fill!(0)
  check(octets.hex == "414243", "Bytes import owns its storage")
  check(sputniktorch.from_bytes("".bytes, device:).numel == 0, "empty Bytes tensor")
  refused = false
  try:
    sputniktorch.from_bytes(octets, shape: [2], device:)
  rescue Sputniktorch.Errors.ShapeError |error|:
    refused = true
  check(refused, "byte payload and shape must agree")

  values = sputniktorch.tensor([[1.0, 3.0], [4.0, 4.0], [5.0, 2.0]], device:, requires_grad: true)
  check(values.argmax(dim: 1).to_a == [1, 0, 0], "argmax tie selects first position")
  check(values.argmax(dim: -1, keepdim: true).shape == [3, 1], "argmax keepdim")
  check(values.argmax().item == 4, "global argmax")
  indices = sputniktorch.tensor([2, 0, 2], dtype: "int64", device:)
  selected = values.index_select(0, indices)
  check(selected.to_a == [[5.0, 2.0], [1.0, 3.0], [5.0, 2.0]], "index_select order and duplicates")
  selected.sum().backward!()
  check(values.grad.to_a == [[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]], "index_select autograd accumulation")

  base = sputniktorch.tensor([[1.0, 2.0], [3.0, 4.0]], device:)
  independent = base.copy
  snapshot = base.snapshot
  base.select(0, 0).fill!(9)
  check(base.to_a == [[9.0, 9.0], [3.0, 4.0]], "view mutation")
  check(independent.to_a == [[1.0, 2.0], [3.0, 4.0]], "copy storage")
  deep = independent.deep_copy()
  deep.zero!
  check(independent.sum().item == 10.0, "generic deep_copy honors native storage hook")
  check(snapshot.to_a == [[1.0, 2.0], [3.0, 4.0]], "snapshot stable")
  denied = false
  try:
    snapshot.detach.select(0, 0).fill!(0)
  rescue Sputniktorch.Errors.ReadOnlyTensorError |e|:
    denied = true
  check(denied, "snapshot alias must be read-only")
  ragged = false
  try:
    sputniktorch.tensor([[1], [2, 3]])
  rescue Sputniktorch.Errors.ShapeError |e|:
    ragged = true
  check(ragged, "ragged input")
  overflow = false
  try:
    sputniktorch.tensor([256], dtype: "uint8")
  rescue Sputniktorch.Errors.DTypeError |e|:
    overflow = true
  check(overflow, "checked dtype conversion")
  check(sputniktorch.tensor(3.0, device:).shape == [], "scalar rank zero")
  check(sputniktorch.zeros([2, 0, 3], device:).to_a == [[], []], "empty shape")
  check(sputniktorch.tensor([2], device:).mul(0.5).item == 1.0, "scalar promotion")
  r1 = sputniktorch.randn([2, 3], seed: 42, device:)
  r2 = sputniktorch.randn([2, 3], seed: 42, device:)
  check(r1.to_a == r2.to_a, "local seeded RNG")

  w = sputniktorch.zeros([1], device:, requires_grad: true)
  optim = sputniktorch.SGD([w, w], lr: 0.05)
  60.times:
    optim.zero_grad!
    w.mul(2).sub(6).square.mean().backward!()
    optim.step!
  check((w.item - 3.0).abs < 0.0001, "train + duplicate parameter deduplication")

  model = sputniktorch.Linear(1, 1, seed: 7, device:)
  optimizer = sputniktorch.SGD(model.parameters, lr: 0.05)
  input = sputniktorch.tensor([[1.0], [2.0], [3.0]], device:)
  target = input.mul(2).add(1)
  saved = model.state_dict
  invalid = saved.copy
  invalid["bias"] = sputniktorch.zeros([2], device:)
  mismatch = false
  try:
    model.load_state_dict!(invalid)
  rescue Sputniktorch.Errors.ShapeError |e|:
    mismatch = true
  check(mismatch, "state dict strict preflight")
  check(model.state_dict["weight"].to_a == saved["weight"].to_a, "failed load preserves model")
  120.times:
    optimizer.zero_grad!
    model.call(input).sub(target).square.mean().backward!()
    optimizer.step!
  trained_loss = model.call(input).sub(target).square.mean().item
  check(trained_loss < 0.01, "Linear training")
  saved_weight = saved["weight"].to_a
  model.load_state_dict!(saved)
  check(model.state_dict["weight"].to_a == saved_weight, "state dict restore")
  optimizer.zero_grad!
  model.call(input).sub(target).square.mean().backward!()
  optimizer.step!
  check(saved["weight"].to_a == saved_weight, "state dict independence")
  check(model.state_dict["weight"].to_a != saved_weight, "optimizer still holds restored parameter identity")
  if device == "mps":
    rejected = false
    try:
      sputniktorch.zeros([1], dtype: "float64", device: "mps")
    rescue Sputniktorch.Errors.DTypeError |e|:
      rejected = true
    check(rejected, "MPS float64 rejection")
    sputniktorch.synchronize()
  print("PASS sputniktorch " + device)
  0
