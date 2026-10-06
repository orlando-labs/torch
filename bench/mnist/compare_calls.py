#!/usr/bin/env python3
"""Optional, sequential MNIST A/B for saved VM/native binaries and Python.

Build the current full-native benchmark first. No builds, tests or samples may
run during measurement. A saved VM uses the current manifest/adapter, isolating
VM runtime changes; the saved native executable contains its original adapter.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from run import ROOT, native_inputs, parse_sample, sha256, source_hashes, validate_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-native", type=Path, required=True)
    parser.add_argument("--before-build-log", type=Path, required=True)
    parser.add_argument("--before-vm", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--processes", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.processes <= 0:
        parser.error("processes must be positive")
    args.epochs, args.batch_size, args.seed, args.threads = 3, 32, 7, 4
    args.train_limit, args.test_limit = 2048, 512
    args.profile, args.execution_scope = False, "run"
    args.loss_atol, args.accuracy_atol = 1e-5, 0.0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sputnik = ROOT.parent / "sputnik-lang/build/sputnik"
    native = ROOT / "build/sputnik/sputniktorch.mnist_benchmark"
    manifest = ROOT / "build/sputniktorch.mnist_benchmark.build.json"
    sources = source_hashes()
    stamp = json.loads((ROOT / "build/mnist-benchmark-build.json").read_text())
    if (not stamp["full_native"] or native_inputs(stamp["sources"]) != native_inputs(sources)
            or stamp["sputnik_sha256"] != sha256(sputnik)
            or stamp["native_sha256"] != sha256(native)):
        raise ValueError("Current native benchmark is stale")
    text = args.before_build_log.read_text()
    before, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
    if (not before["native_graph_full_coverage"] or before["native_bytecode_fallback"]
            or before["native_hash"] != sha256(args.before_native)):
        raise ValueError("Saved native binary/coverage does not match its build log")
    commands = {"native-before": [str(args.before_native.resolve())],
                "sputnik-built": [str(native)],
                "sputnik-vm": [str(sputnik), "run", str(manifest), "--grant", "ffi",
                             "--grant", "fs.read", "--"],
                "python": [args.python, str(ROOT / "bench/mnist/python_mnist.py")]}
    if args.before_vm:
        commands["vm-before"] = [str(args.before_vm.resolve()), "run", str(manifest),
                                 "--grant", "ffi", "--grant", "fs.read", "--"]
    hashes = {lane: sha256(command[0]) for lane, command in commands.items()}
    data = ROOT / "build/datasets/mnist"
    dataset_hashes = {name: sha256(data / name) for name in
                      ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                       "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTORCH_ENABLE_MPS_FALLBACK="0",
               SPUTNIK_ROOT=str(ROOT.parent / "sputnik-lang"),
               SPUTNIK_NATIVE_RUNTIME_ROOT=str(ROOT.parent / "sputnik-lang"))
    for key in ("AMT_BOUNDARY_PROFILE", "SPUTNIK_HEAP_STATS", "DYLD_PRINT_LIBRARIES"):
        env.pop(key, None)
    info = json.loads(subprocess.check_output([args.python, "-c", "import json,torch; "
        "print(json.dumps(dict(version=torch.__version__,mps=torch.backends.mps.is_available())))"],
        env=env, text=True))
    lock = json.loads((ROOT / "build/native-lock.json").read_text())
    if info["version"] != lock["torch_version"] or (args.device == "mps" and not info["mps"]):
        raise ValueError("LibTorch/Python version mismatch or unavailable MPS")
    envelope = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_sha256=sources, executables_sha256=hashes, native_build=stamp,
                    before_build=before, python_environment=info,
                    dataset_sha256=dataset_hashes,
                    libtorch_library_sha256={name: sha256(Path(lock["torch_prefix"]) / "lib" / name)
                        for name in ("libtorch.dylib", "libtorch_cpu.dylib", "libc10.dylib")}, samples=[])
    common = ["--device", args.device, "--epochs", "3", "--batch-size", "32",
              "--seed", "7", "--train-limit", "2048", "--test-limit", "512",
              "--data", str(data)]
    for process in range(-1, args.processes):
        names = list(commands)
        rotation = (process + 1) % len(names)
        names = names[rotation:] + names[:rotation]
        current = []
        for lane in names:
            tag = "warmup" if process < 0 else str(process)
            command = commands[lane] + common + ["--checkpoint", str(output / f"{tag}-{lane}.pt")]
            command += (["--optimizer", "packed"] if lane == "python"
                        else ["--execution-scope", "run"])
            print(tag, lane, flush=True)
            started = time.perf_counter()
            result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                    capture_output=True, timeout=600)
            wall = time.perf_counter() - started
            (output / f"{tag}-{lane}.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"{lane}: {result.stderr[-1500:]}")
            sample = parse_sample(result.stdout, wall)
            sample.update(lane=lane, repeat=process, warmup=process < 0, command=command)
            current.append(sample)
            envelope["samples"].append(sample)
            (output / "raw.json").write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n")
        validate_samples(current, args)
    if sources != source_hashes() or hashes != {lane: sha256(cmd[0]) for lane, cmd in commands.items()}:
        raise RuntimeError("Sources or executables changed during measurement")
    if dataset_hashes != {name: sha256(data / name) for name in dataset_hashes}:
        raise RuntimeError("Dataset changed during measurement")
    summary = {lane: statistics.median(s["loop_seconds"] for s in envelope["samples"]
                                      if s["lane"] == lane and not s["warmup"]) for lane in commands}
    pairs = [{s["lane"]: s["loop_seconds"] for s in envelope["samples"] if s["repeat"] == i}
             for i in range(args.processes)]
    envelope.update(summary_loop_seconds=summary, paired_loop_seconds=pairs)
    (output / "results.json").write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("native before/after ratios", [p["sputnik-built"] / p["native-before"] for p in pairs])
    print("native/Python ratios", [p["sputnik-built"] / p["python"] for p in pairs])


if __name__ == "__main__":
    main()
