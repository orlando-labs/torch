package sputniktorch.mnist.benchmark_options

export BenchmarkOptions

class BenchmarkOptions:
  attr data
  attr checkpoint
  attr device
  attr epochs
  attr batch_size
  attr seed
  attr train_limit
  attr test_limit
  attr profile
  attr execution_scope

  def init():
    parser = ArgParser(name: "MNIST benchmark")
    parser.arg("--data", default: "build/datasets/mnist")
    parser.arg("--checkpoint", default: "build/mnist-benchmark.pt")
    parser.arg("--device", choices: ["cpu", "mps"], default: "mps")
    parser.arg("--epochs", type: Int, default: 5)
    parser.arg("--batch-size", type: Int, default: 128)
    parser.arg("--seed", type: Int, default: 7)
    parser.arg("--train-limit", type: Int)
    parser.arg("--test-limit", type: Int)
    parser.flag("--profile")
    parser.arg("--execution-scope", choices: ["run", "epoch"], default: "run")
    args = parser.parse_or_raise
    @data = args["data"]
    @checkpoint = args["checkpoint"]
    @device = args["device"]
    @epochs = args["epochs"]
    @batch_size = args["batch_size"]
    @seed = args["seed"]
    @train_limit = args["train_limit"]
    @test_limit = args["test_limit"]
    @profile = args["profile"]
    @execution_scope = args["execution_scope"]
    raise ValueError("epochs and batch size must be positive") if @epochs <= 0 or @batch_size <= 0
