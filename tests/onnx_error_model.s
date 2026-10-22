package torch.onnx_error_model

import torch
from torch.nn import Module, Identity
from torch.nn.functional import kernel
from torch.autograd import grad
from torch.onnx import export_model

export ErrorModel

class ErrorModel < Module:
  def init(@operation):
    _initialize_registry!
    @identity = Identity()
    @identity.eval!
    @weight = torch.ones([1], requires_grad: true)
    register_parameter!("weight", @weight)
    @saved = state_dict

  def forward(x):
    case! @operation.str:
      when "item": x.sum().item
      when "mutation": x.zero!
      when "backward": x.sum().backward!()
      when "grad": grad([x.sum()], [x])
      when "state":
        load_state_dict!(@saved)
        x
      when "unsupported": kernel("pixel_shuffle", [x], [2])
      when "nested": export_model(@identity, x)
      else: raise ValueError("unknown error case")
