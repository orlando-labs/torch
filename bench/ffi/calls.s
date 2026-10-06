package sputniktorch.ffi_calls

from sputniktorch.backends import execution_batch, synchronize
from sputniktorch.autograd.context import execution_mode
from sputniktorch.tensor.factories import ones
from sputniktorch.tensor.value import Tensor

export main

native def abi_constant() from "amt_bench_constant"
native def abi_identity(value) from "amt_bench_identity"
native def abi_identity4(first, second, third, fourth) from "amt_bench_identity"
native def abi_int(value) from "amt_bench_int"
native def abi_handle(value) from "amt_bench_handle"
native def abi_numel(value) from "amt_bench_numel"
native def abi_string(value) from "amt_bench_string"
native def abi_list_len(value) from "amt_bench_list_len"
native def abi_list_sum(value) from "amt_bench_list_sum"
native def abi_relu(value) from "amt_bench_relu"
native def abi_generic_relu(value, op as Str, args, mode as Int) from "amt_unary"
native def cpp_loop(op as Str, core, count as Int) from "amt_bench_cpp_loop"

def constant(): 42

def identity(value): value

def identity4(first, second, third, fourth): first

class Identity:
  def constant(): 42
  def identity(value): value

def elapsed(started): (Time.monotonic - started).total_nanoseconds.float / 1e9

def sync(device): synchronize() if device == "mps"

def checksum(name, value):
  if name == "loop_handle" or name == "sputnik_identity_handle" or name == "abi_identity_handle" or name == "abi_handle_checked" or name == "relu_specialized_core" or name == "relu_generic_core" or name == "cpp_relu":
    Tensor(value).sum.item
  else if Tensor === value:
    value.sum.item
  else:
    value

# Every ordinary row includes the same loop + captured block invocation.
# C++ batch controls have their own C++ loop and one ABI call for the batch.
def measure(name, iterations, repeats, device, selected, batched: false, &block):
  return null if selected != "all" and selected != name
  rows = []
  value = null
  (repeats + 1).times |repeat|:
    execution_batch:
      sync(device)
      started = Time.monotonic
      if batched:
        value = block()
      else:
        iterations.times:
          value = block()
      sync(device)
      seconds = elapsed(started)
      rows.push!(seconds) if repeat > 0
  print "FFI_ATOM #{Json.generate({name:, iterations:, seconds: rows, checksum: checksum(name, value)})}"

def main():
  parser = ArgParser(name: "FFI call decomposition")
  parser.arg("--device", choices: ["cpu", "mps"], default: "cpu")
  parser.arg("--iterations", type: Int, default: 300000)
  parser.arg("--kernel-iterations", type: Int, default: 3000)
  parser.arg("--repeats", type: Int, default: 5)
  parser.arg("--case", default: "all")
  parser.arg("--size", type: Int, default: 25088)
  args = parser.parse_or_raise
  n = args["iterations"]
  k = args["kernel_iterations"]
  repeats = args["repeats"]
  size = args["size"]
  selected = args["case"]
  device = args["device"]
  raise ValueError("counts and size must be positive") if n <= 0 or k <= 0 or repeats <= 0 or size <= 0
  x = ones([size], device:)
  core = x._core
  receiver = Identity()
  small = [1, 2, 3, 4]
  large = Array.of(32): $it
  text = "relu"
  long_text = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  # Keep blocks here rather than add a forwarding helper around each row.
  measure("loop_scalar", n, repeats, "cpu", selected): 42
  measure("loop_handle", n, repeats, "cpu", selected): core
  measure("sputnik_constant", n, repeats, "cpu", selected): constant()
  measure("sputnik_identity_int", n, repeats, "cpu", selected): identity(42)
  measure("sputnik_identity4_int", n, repeats, "cpu", selected): identity4(42, 1, 2, 3)
  measure("sputnik_identity_handle", n, repeats, "cpu", selected): identity(core)
  measure("sputnik_method_constant", n, repeats, "cpu", selected): receiver.constant
  measure("sputnik_method_identity", n, repeats, "cpu", selected): receiver.identity(42)
  measure("abi_constant", n, repeats, "cpu", selected): abi_constant()
  measure("abi_identity_int", n, repeats, "cpu", selected): abi_identity(42)
  measure("abi_identity4_int", n, repeats, "cpu", selected): abi_identity4(42, 1, 2, 3)
  measure("abi_identity_handle", n, repeats, "cpu", selected): abi_identity(core)
  measure("abi_int_rebox", n, repeats, "cpu", selected): abi_int(42)
  measure("abi_handle_checked", n, repeats, "cpu", selected): abi_handle(core)
  measure("abi_numel_free", n, repeats, "cpu", selected): abi_numel(core)
  measure("core_numel_method", n, repeats, "cpu", selected): core.numel
  measure("tensor_numel", n, repeats, "cpu", selected): x.numel
  measure("execution_mode", n, repeats, "cpu", selected): execution_mode()
  measure("abi_string_4", n, repeats, "cpu", selected): abi_string(text)
  measure("abi_string_256", n, repeats, "cpu", selected): abi_string(long_text)
  measure("abi_list_len_4", n, repeats, "cpu", selected): abi_list_len(small)
  measure("abi_list_sum_4", n, repeats, "cpu", selected): abi_list_sum(small)
  measure("abi_list_sum_32", n, repeats, "cpu", selected): abi_list_sum(large)
  measure("relu_specialized_core", k, repeats, device, selected): abi_relu(core)
  measure("relu_generic_core", k, repeats, device, selected): abi_generic_relu(core, "relu", null, 0)
  measure("relu_generic_tensor", k, repeats, device, selected): Tensor(abi_generic_relu(core, "relu", null, 0))
  measure("relu_public", k, repeats, device, selected): x.relu
  measure("cpp_constant", n, repeats, "cpu", selected, batched: true): cpp_loop("constant", core, n)
  measure("cpp_identity_int", n, repeats, "cpu", selected, batched: true): cpp_loop("identity", core, n)
  measure("cpp_numel", n, repeats, "cpu", selected, batched: true): cpp_loop("numel", core, n)
  measure("cpp_relu", k, repeats, device, selected, batched: true): cpp_loop("relu", core, k)
  0
