package sputniktorch.onnx_operator_model

import sputniktorch
import task
from sputniktorch.nn import Module
from sputniktorch.nn.functional import scaled_dot_product_attention

export OperatorModel

class OperatorModel < Module:
  def init(@operation): _initialize_registry!

  def forward(x, mask: null):
    case! @operation.str:
      when "math":
        y = (x + 1.5).clamp(min: -3.0, max: 3.0)
        (y.square.mean(dim: 1), y.sin + y.cos, y.argmax(dim: 1), y.sum())
      when "layout":
        y = x.unsqueeze(1).squeeze(dim: 1).transpose(1, 2)
        (x.flatten(), x.flatten(start_dim: 0, end_dim: 1), y,
          x.narrow(0, -1, 1), sputniktorch.cat([x, x], dim: 1), sputniktorch.stack([x, x], dim: 1))
      when "types":
        y = x * 0.5
        (y, x.gt(0.5), x.eq(1), x.argmax(dim: 1), x.to(dtype: "float32", device: x.device))
      when "attention":
        scaled_dot_product_attention(x, x, x, mask:)
      when "causal":
        scaled_dot_product_attention(x, x, x, mask:, causal: true)
      when "task":
        child = task.spawn: x.square + 1
        child.wait
      else: raise ValueError("unknown test operation")
