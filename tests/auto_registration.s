package sputniktorch.auto_registration_test

from sputniktorch.nn import Module, Linear, Sequential, ModuleList, ReLU, BatchNorm1d
from sputniktorch.nn import MultiheadAttention, RNNCell, Dropout
from sputniktorch.tensor.factories import ones
from sputniktorch.checks import check

export main, run

class AutoModel < Module:
  def init(device):
    @features = Sequential([BatchNorm1d(2, device:), ReLU()])
    @head = Linear(2, 1, device:)

  def forward(x): @head(@features(x))
  def forward!(x): @head(@features.call!(x))

class AliasModel < Module:
  def init(device):
    _initialize_registry!
    @projection = Linear(2, 1, device:)
    register_module!("dense", @projection)

class ContainerModel < Module:
  def init(device): @layers = ModuleList([Linear(2, 1, device:)])

class ArrayModel < Module:
  def init(device): @layers = [Linear(2, 1, device:)]

class SharedModel < Module:
  def init(device):
    @first = Linear(2, 1, device:)
    @second = @first

class EvalModel < Module:
  def init():
    _initialize_registry!
    eval!
    @drop = Dropout()

class CycleModel < Module:
  def init(): @child = self

def run(device):
  model = AutoModel(device)
  names = model.named_parameters.keys.sorted
  check(names == ["features.0.bias", "features.0.weight", "head.bias", "head.weight"], "automatic parameter paths")
  check(model.named_buffers.count == 3 and model.stateful?, "automatic buffers and stateful traversal")
  input = ones([2, 2], device:, requires_grad: true)
  model.call!(input).sum().backward!
  check(model.parameters.all?: $it.grad != null, "automatic children participate in training")
  saved = model.state_dict
  model.eval!
  check(not model.stateful?, "eval propagates to discovered children")
  check(model(input).shape == [2, 1], "evaluation forward")
  model.load_state_dict!(saved)
  model.after_init!
  check(model.named_parameters.keys.sorted == names, "registry discovery is idempotent")
  check(AliasModel(device).named_parameters.keys.sorted == ["dense.bias", "dense.weight"], "explicit aliases take precedence")
  check(ContainerModel(device).parameters.count == 2, "automatic ModuleList field")
  check(ArrayModel(device).parameters.empty?, "ordinary arrays are not model registries")
  check(SharedModel(device).named_parameters.count == 4, "shared children retain both automatic field paths")
  check(not EvalModel().instance_fields["drop"].training?, "discovered children inherit existing eval mode")
  check(MultiheadAttention(4, 2, device:).state_dict.has_key?("query.weight"), "attention checkpoint alias is preserved")
  check(RNNCell(2, 1, device:).state_dict.has_key?("input.weight"), "recurrent checkpoint alias is preserved")
  rejected = false
  try:
    CycleModel()
  rescue ValueError |error|:
    rejected = true
  check(rejected, "after_init rejects cycles at the constructor call site")
  print "PASS automatic module registration on #{device}"
  0

def main(): run("cpu")
