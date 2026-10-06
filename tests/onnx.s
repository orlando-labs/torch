package sputniktorch.onnx_test

import sputniktorch
from sputniktorch.onnx import export_model
from sputniktorch.autograd import grad_enabled?
from sputniktorch.mnist.network import SmallResNet
from sputniktorch.checks import check

export main

def main():
  model = SmallResNet(seed: 7)
  model.eval!
  sample = sputniktorch.randn([2, 1, 28, 28], seed: 17)
  program = export_model(model, sample, path: "build/mnist.onnx",
    input_names: ["images"], output_names: ["logits"],
    dynamic_axes: {"images": {0: "batch"}, "logits": {0: "batch"}})
  check(program.opset_version == 18 and program.node_count > 10, "ONNX graph metadata")
  check(fs.read_bytes("build/mnist.onnx").hex == program.bytes.hex, "ONNX save bytes")
  check(grad_enabled?(), "ONNX restores grad context")
  references = {}
  [1, 3].each |batch|:
    input = sputniktorch.randn([batch, 1, 28, 28], seed: batch + 17)
    references["images.#{batch}"] = input
    references["logits.#{batch}"] = model(input)
  sputniktorch.save(references, "build/mnist-onnx-reference.pt")
  print "PASS ONNX MNIST export | #{program.node_count} nodes | #{program.byte_count} bytes"
  0
