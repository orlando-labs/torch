package torch.distributed.ddp

from torch.errors import Error

export DDP
native def _parameters!(group, entries, operation) from "amt_ddp_parameters" blocking

# Explicit post-backward reduction; equal local batch sizes, CPU/Gloo only.
class DDP:
  def init(model, group):
    @model = model
    @group = group
    @ready = false

  def _entries(): @model.named_parameters.map |name, parameter|: [name, parameter._core]

  def synchronize!():
    _parameters!(@group._core, _entries, "sync")
    @ready = true
    self

  def call(x):
    if not @ready:
      raise ValueError("call synchronize! before DDP forward")
    @model.call(x)

  def backward!(loss):
    if not @ready:
      raise ValueError("call synchronize! before DDP backward")
    loss.backward!()
    _parameters!(@group._core, _entries, "reduce")
    self

  def parameters(): @model.parameters
  def state_dict(): @model.state_dict
