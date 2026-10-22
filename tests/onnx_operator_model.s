package torch.onnx_operator_model

import torch
import task
from torch.nn import Module
from torch.nn.functional import scaled_dot_product_attention
from torch.backends import execution_batch

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
          x.narrow(0, -1, 1), torch.cat([x, x], dim: 1), torch.stack([x, x], dim: 1))
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
      when "batch":
        execution_batch(): x.relu.square + 1
      else: raise ValueError("unknown test operation")
