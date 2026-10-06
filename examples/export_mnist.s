package sputniktorch.export_mnist_example

import sputniktorch
from sputniktorch.mnist.network import SmallResNet
from sputniktorch.mnist.export_options import ExportOptions
from sputniktorch.onnx import export_model

export main

def main():
  options = ExportOptions()
  device = options.device
  model = SmallResNet(device:)
  model.load_state_dict!(sputniktorch.load(options.checkpoint, map_location: device))
  model.eval!
  example = sputniktorch.randn([2, 1, 28, 28], seed: 17, device:)
  program = export_model(model, example, path: options.output,
    input_names: ["images"], output_names: ["logits"],
    dynamic_axes: {"images": {0: "batch"}, "logits": {0: "batch"}})
  references = {}
  sputniktorch.no_grad:
    [1, 3].each |batch|:
      images = sputniktorch.randn([batch, 1, 28, 28], seed: batch + 17, device:)
      references["images.#{batch}"] = images
      references["logits.#{batch}"] = model(images)
  sputniktorch.save(references, options.references)
  print "Exported trained SmallResNet on #{device} | #{program.node_count} nodes | #{program.byte_count} bytes"
  0
