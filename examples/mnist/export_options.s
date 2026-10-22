package torch.mnist.export_options

export ExportOptions

class ExportOptions:
  attr checkpoint
  attr device
  attr output
  attr references

  def init():
    parser = ArgParser(name: "torch ONNX", about: "Export a trained MNIST SmallResNet")
    parser.arg("--checkpoint", default: "build/mnist-resnet.pt")
    parser.arg("--device", choices: ["cpu", "mps"], default: "cpu")
    parser.arg("--output", default: "build/mnist-trained.onnx")
    parser.arg("--references", default: "build/mnist-trained-reference.pt")
    args = parser.parse_or_raise
    @checkpoint = args["checkpoint"]
    @device = args["device"]
    @output = args["output"]
    @references = args["references"]
