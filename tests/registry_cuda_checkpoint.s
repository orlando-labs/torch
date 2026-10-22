from slowpilot.torch.tensor.factories import tensor
from slowpilot.torch.serialization.state_dict import load, save

parser = ArgParser(name: "CUDA checkpoint round trip")
parser.arg("--path", required: true)
args = parser.parse_or_raise()
weight = tensor([[1.0, 2.0], [3.0, 4.0]], device: "cuda:0")
save({"weight": weight}, args[:path])
restored = load(args[:path], map_location: "cuda:0")["weight"]
raise ValueError("Checkpoint did not restore to CUDA") unless restored.device.type == "cuda"
raise ValueError("Checkpoint data changed") unless restored.to_a == weight.to_a
print "PASS CUDA checkpoint round trip"
