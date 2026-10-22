#!/usr/bin/env python3
"""Build once, validate and compare Sputnik VM/native and the optional Python example."""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANES = ("sputnik-vm", "sputnik-built", "python")
PREFIX = "MNIST_BENCH "


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes():
    files = [ROOT / "tools/build", ROOT / "cmake/BuildSputnik.cmake", ROOT / "CMakeLists.txt"]
    for directory, patterns in (("src", ("*.s",)), ("native", ("*.cpp", "*.h")),
                                ("examples/mnist", ("*.s",)), ("tests", ("*.s",)),
                                ("bench/mnist", ("*.s", "*.py"))):
        for pattern in patterns:
            files.extend((ROOT / directory).rglob(pattern))
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(files))}


def native_inputs(sources):
    return {key: value for key, value in sources.items() if not key.endswith(".py")}


def command_result(command, env, timeout, log):
    started = time.perf_counter()
    try:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        log.write_text(str(error), encoding="utf-8")
        raise RuntimeError(f"Timed out; see {log}") from error
    wall = time.perf_counter() - started
    log.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Exit {result.returncode}: {' '.join(map(str, command))}; see {log}\n"
                           + result.stderr[-2000:] + result.stdout[-2000:])
    return result.stdout, wall


def parse_sample(stdout, wall):
    records = [line[len(PREFIX):] for line in stdout.splitlines() if line.startswith(PREFIX)]
    if len(records) != 1:
        raise ValueError("Expected exactly one MNIST_BENCH result")
    sample = json.loads(records[0])
    if sample.get("schema") != "torch.mnist.sample.v1" or not sample.get("epochs"):
        raise ValueError("Invalid MNIST sample schema or empty epochs")
    sample["wall_seconds"] = wall
    # Includes imports, VM manifest/cache checks, loading, result printing and teardown.
    sample["startup_teardown_seconds"] = wall - sample["workload_seconds"]
    sample["train_seconds"] = sum(e["train_seconds"] for e in sample["epochs"])
    sample["eval_seconds"] = sum(e["eval_seconds"] for e in sample["epochs"])
    sample["loop_seconds"] = sample["train_seconds"] + sample["eval_seconds"]
    sample["cold_train_seconds"] = sample["epochs"][0]["train_seconds"]
    sample["steady_train_seconds"] = (statistics.mean(e["train_seconds"] for e in sample["epochs"][1:])
                                      if len(sample["epochs"]) > 1 else None)
    for key in ("wall_seconds", "workload_seconds", "data_seconds", "setup_seconds", "save_seconds",
                "restore_seconds", "restore_eval_seconds", "train_seconds", "eval_seconds"):
        if not math.isfinite(sample[key]) or sample[key] < 0:
            raise ValueError(f"Invalid timing: {key}")
    for epoch in sample["epochs"]:
        if any(not math.isfinite(epoch[key]) for key in ("train_loss", "test_loss", "accuracy")):
            raise ValueError("Non-finite training metric")
    if abs(sample["checkpoint_loss"] - sample["epochs"][-1]["test_loss"]) > 1e-7:
        raise ValueError("Checkpoint changed loss")
    if sample["checkpoint_accuracy"] != sample["epochs"][-1]["accuracy"]:
        raise ValueError("Checkpoint changed accuracy")
    return sample


def validate_samples(samples, args):
    expected_counts = (args.train_limit or 60000, args.test_limit or 10000)
    for sample in samples:
        if (sample["train_count"], sample["test_count"]) != expected_counts:
            raise ValueError("Sample split sizes do not match the requested workload")
        if (sample["parameters"] != 19706 or sample["seed"] != args.seed or
                sample["batch_size"] != args.batch_size or sample["device"] != args.device or
                sample["profiled"] != args.profile or len(sample["epochs"]) != args.epochs or
                sample["threads"] != args.threads):
            raise ValueError("Sample configuration does not match the requested workload")
        if (sample["lane"] != "python" and
                sample.get("execution_scope", "epoch") != args.execution_scope):
            raise ValueError("Sputnik execution scope does not match the requested workload")
    reference = next(sample for sample in samples if sample["lane"] == "python")
    for sample in samples:
        if (sample["version"] != reference["version"] or
                sample["interop_threads"] != reference["interop_threads"]):
            raise ValueError("PyTorch/LibTorch versions or interop threads differ")
        for epoch, expected in zip(sample["epochs"], reference["epochs"]):
            for key in ("train_loss", "test_loss", "accuracy"):
                tolerance = args.accuracy_atol if key == "accuracy" else args.loss_atol
                if abs(epoch[key] - expected[key]) > tolerance:
                    raise ValueError(f"Numerical mismatch: {sample['lane']} epoch {epoch['epoch']} {key}: "
                                     f"{epoch[key]} vs Python {expected[key]} (atol={tolerance})")


def stats(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return dict(n=len(values), median=statistics.median(values), mean=mean, min=min(values),
                max=max(values), stdev=sd, cv=sd / mean if mean else 0.0)


def summarize(samples):
    result = {}
    fields = ("wall_seconds", "startup_teardown_seconds", "workload_seconds", "data_seconds",
              "setup_seconds", "train_seconds", "eval_seconds", "loop_seconds", "cold_train_seconds",
              "steady_train_seconds", "save_seconds", "restore_seconds", "restore_eval_seconds")
    for lane in LANES:
        group = [s for s in samples if s["lane"] == lane and not s["warmup"]]
        result[lane] = {key: stats([s[key] for s in group]) for key in fields if group[0][key] is not None}
        result[lane]["train_images_per_second"] = stats(
            [s["train_count"] * len(s["epochs"]) / s["train_seconds"] for s in group])
        profiles = [s for s in group if s["profiled"]]
        if profiles:
            result[lane]["profile"] = {
                key: stats([sum(e["profile"][key] for e in s["epochs"]) for s in profiles])
                for key in profiles[0]["epochs"][0]["profile"]}
    return result


def compare_baseline(envelope, baseline, threshold):
    keys = ("device", "epochs", "batch_size", "seed", "train_limit", "test_limit", "threads",
            "profile", "python_optimizer")
    for key in keys:
        if envelope["config"][key] != baseline["config"][key]:
            raise ValueError(f"Baseline has a different workload: {key}")
    if envelope["config"]["execution_scope"] != baseline["config"].get("execution_scope", "epoch"):
        raise ValueError("Baseline has a different Sputnik execution scope")
    for key in ("machine", "processor", "torch_version"):
        if envelope["host"][key] != baseline["host"][key]:
            raise ValueError(f"Baseline has a different host/backend: {key}")
    changes = []
    for lane in LANES:
        for phase, current in envelope["summary"][lane].items():
            if phase == "profile":
                continue
            previous = baseline["summary"][lane][phase]["median"]
            ratio = current["median"] / previous if previous else 1.0
            slower = ratio < 1 / (1 + threshold) if phase.endswith("per_second") else ratio > 1 + threshold
            changes.append(dict(lane=lane, phase=phase, ratio=ratio, regression=slower))
        for phase, current in envelope["summary"][lane].get("profile", {}).items():
            previous = baseline["summary"][lane]["profile"][phase]["median"]
            ratio = current["median"] / previous if previous else 1.0
            changes.append(dict(lane=lane, phase="profile." + phase, ratio=ratio,
                                regression=ratio > 1 + threshold))
    return changes


def check_python_speed(samples, minimum_speedup):
    measured = [sample for sample in samples if not sample["warmup"]]
    reference = {sample["repeat"]: sample["loop_seconds"]
                 for sample in measured if sample["lane"] == "python"}
    comparisons = []
    for sample in measured:
        if sample["lane"] == "python":
            continue
        python_seconds = reference[sample["repeat"]]
        if python_seconds <= 0:
            raise ValueError("Python reference loop must have positive duration")
        ratio = sample["loop_seconds"] / python_seconds
        comparisons.append(dict(lane=sample["lane"], repeat=sample["repeat"],
                                time_ratio=ratio, passed=ratio < 1 / minimum_speedup))
    return dict(phase="loop_seconds", minimum_speedup=minimum_speedup,
                passed=all(row["passed"] for row in comparisons), comparisons=comparisons)


def report(envelope):
    c, summary = envelope["config"], envelope["summary"]
    mode = "PROFILE (synchronization after every stage)" if c["profile"] else "THROUGHPUT"
    lines = [f"# MNIST — {c['device']} — {mode}", "", f"UTC: {envelope['created_at']}", "",
             f"SmallResNet: 19,706 parameters; train {c['train_limit'] or 60000}, "
             f"test {c['test_limit'] or 10000}; batch {c['batch_size']}; epochs {c['epochs']}; seed {c['seed']}.",
             f"Measured repeats: {c['repeats']}; process warmups: {c['warmups']}; "
             f"threads: {c['threads']}; Python optimizer: {c['python_optimizer']}; "
             f"Sputnik execution scope: {c['execution_scope']}.", "",
             "Times are medians in seconds. Compilation and VM cache preparation are excluded.", "",
             "| Phase | sputnik-vm | sputnik-built | python | VM / Python | built / Python |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, label in (("wall_seconds", "Process wall"), ("startup_teardown_seconds", "Startup + teardown"),
                       ("data_seconds", "IDX + normalization"), ("setup_seconds", "Model + optimizer setup"),
                       ("train_seconds", "Train, all epochs"), ("eval_seconds", "Evaluation, all epochs"),
                       ("loop_seconds", "Train + evaluation"), ("cold_train_seconds", "First train epoch"),
                       ("steady_train_seconds", "Later train epoch, mean per process"),
                       ("save_seconds", "Checkpoint save"), ("restore_seconds", "Model recreation + load"),
                       ("restore_eval_seconds", "Checkpoint evaluation")):
        if key not in summary["python"]:
            continue
        values = [summary[lane][key]["median"] for lane in LANES]
        lines.append(f"| {label} | " + " | ".join(f"{v:.6f}" for v in values) +
                     f" | {values[0] / values[2]:.2f}× | {values[1] / values[2]:.2f}× |")
    lines += ["", "Ratios above 1 mean more time than Python.", "",
              "| Lane | Train images/s | Loop min–max, s | Loop CV | Final test loss | Accuracy |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for lane in LANES:
        sample = next(s for s in envelope["samples"] if s["lane"] == lane and not s["warmup"])
        loop, last = summary[lane]["loop_seconds"], sample["epochs"][-1]
        lines.append(f"| {lane} | {summary[lane]['train_images_per_second']['median']:.1f} | "
                     f"{loop['min']:.3f}–{loop['max']:.3f} | {loop['cv']:.1%} | "
                     f"{last['test_loss']:.8f} | {last['accuracy']:.2%} |")
    if c["profile"]:
        lines += ["", "Training stage totals (profile instrumentation adds overhead):", "",
                  "| Stage | sputnik-vm | sputnik-built | python |", "| --- | ---: | ---: | ---: |"]
        for key in summary["python"]["profile"]:
            lines.append(f"| {key} | " + " | ".join(f"{summary[lane]['profile'][key]['median']:.6f}"
                                                  for lane in LANES) + " |")
    lines += ["", "Numerical checks passed for every epoch and checkpoint round trip.", "",
              "Host and exact source/executable/dataset hashes, raw samples and run order are in results.json.",
              "Process wall includes Python import and cached VM manifest/extension checks. "
              "The VM executes bytecode through sputnik run; the built lane requires full native coverage.",
              "Checkpoint times compare the two implementations of compatible state-dict serialization."]
    for change in envelope.get("baseline_comparison", []):
        if change["regression"]:
            lines.append(f"\nRegression candidate: {change['lane']} {change['phase']} "
                         f"{change['ratio']:.2f}× baseline.")
    if "python_speed_check" in envelope:
        check = envelope["python_speed_check"]
        lines += ["", f"Python speed check: {'PASS' if check['passed'] else 'FAIL'}; "
                  f"requires >{check['minimum_speedup']:.3f}× speedup in every measured pair.", "",
                  "| Lane | Passing pairs | Worst time ratio to Python |",
                  "| --- | ---: | ---: |"]
        for lane in LANES[:-1]:
            rows = [row for row in check["comparisons"] if row["lane"] == lane]
            lines.append(f"| {lane} | {sum(row['passed'] for row in rows)}/{len(rows)} | "
                         f"{max(row['time_ratio'] for row in rows):.3f}× |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--python-optimizer", choices=("matched", "packed", "torch"), default="matched")
    parser.add_argument("--execution-scope", choices=("run", "epoch"), default="run")
    parser.add_argument("--sputnik-root", type=Path, default=ROOT.parent / "sputnik-lang")
    parser.add_argument("--data", type=Path, default=ROOT / "build/datasets/mnist")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--loss-atol", type=float, default=1e-5)
    parser.add_argument("--accuracy-atol", type=float, default=0.0)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--regression-threshold", type=float, default=0.15)
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--require-faster-than-python", action="store_true")
    parser.add_argument("--minimum-speedup", type=float, default=1.0)
    args = parser.parse_args()
    for key in ("epochs", "batch_size", "repeats", "threads", "timeout"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.warmups < 0 or min(args.loss_atol, args.accuracy_atol, args.regression_threshold) < 0:
        parser.error("warmups and tolerances must be nonnegative")
    for key, maximum in (("train_limit", 60000), ("test_limit", 10000)):
        if getattr(args, key) is not None and not 1 <= getattr(args, key) <= maximum:
            parser.error(f"--{key.replace('_', '-')} must be within the MNIST split")
    if args.fail_on_regression and not args.baseline:
        parser.error("--fail-on-regression requires --baseline")
    if not math.isfinite(args.minimum_speedup) or args.minimum_speedup < 1:
        parser.error("--minimum-speedup must be finite and at least 1")
    if args.profile and args.require_faster_than_python:
        parser.error("--require-faster-than-python requires an unprofiled throughput run")
    sputnik = args.sputnik_root.resolve() / "build/sputnik"
    python = shutil.which(args.python) or args.python
    output = args.output or ROOT / "build/bench/mnist" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               PYTORCH_ENABLE_MPS_FALLBACK="0", SPUTNIK_ROOT=str(args.sputnik_root.resolve()))
    # Debug/profiling environment inherited from a shell would skew the baseline.
    for key in ("SPUTNIK_HEAP_STATS", "DYLD_PRINT_LIBRARIES", "AMT_BOUNDARY_PROFILE"):
        env.pop(key, None)
    sources = source_hashes()
    build_stamp = ROOT / "build/mnist-benchmark-build.json"
    native = ROOT / "build/sputnik/torch.mnist_benchmark"
    manifest = ROOT / "build/torch.mnist_benchmark.build.json"
    compiler_hash = sha256(sputnik)
    if args.skip_build:
        stamp = json.loads(build_stamp.read_text())
        if (native_inputs(stamp["sources"]) != native_inputs(sources) or
                stamp["sputnik_sha256"] != sha256(sputnik) or stamp["native_sha256"] != sha256(native)):
            raise ValueError("Stale benchmark executable; rerun without --skip-build")
    else:
        print("Building full native benchmark (outside measurements)…", flush=True)
        stdout, _ = command_result([ROOT / "tools/build", "bench/mnist/benchmark.s"], env,
                                  args.timeout, output / "build.log")
        build, _ = json.JSONDecoder().raw_decode(stdout[stdout.index("{"):])
        if not build["native_graph_full_coverage"] or build["native_bytecode_fallback"]:
            raise ValueError("Benchmark native executable has VM fallback")
        if native_inputs(source_hashes()) != native_inputs(sources):
            raise ValueError("Sputnik/native sources changed during the build; rerun the benchmark")
        if sha256(sputnik) != compiler_hash:
            raise ValueError("Sputnik compiler changed during the build; rerun the benchmark")
        stamp = dict(sources=native_inputs(sources), sputnik_sha256=compiler_hash, native_sha256=sha256(native),
                     full_native=True, native_backend=build["native_backend"])
        build_stamp.write_text(json.dumps(stamp, indent=2) + "\n")
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config["python"] = str(python)
    config["data"] = str(args.data.resolve())
    dataset_hashes = {name: sha256(args.data / name) for name in
                      ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                       "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")}
    lock = json.loads((ROOT / "build/native-lock.json").read_text())
    processor = platform.processor()
    if sys.platform == "darwin":
        cpu_info = subprocess.run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                                  text=True, capture_output=True)
        if cpu_info.returncode == 0:
            processor = cpu_info.stdout.strip()
    metadata, _ = command_result([python, "-c",
        "import json,sys,torch; print(json.dumps(dict(python=sys.version, executable=sys.executable, "
        "torch_path=torch.__file__, torch_version=torch.__version__, config=torch.__config__.show(), "
        "parallel=torch.__config__.parallel_info(), mps_available=torch.backends.mps.is_available())))"],
        env, args.timeout, output / "python-environment.log")
    python_environment = json.loads(metadata)
    if python_environment["torch_version"] != lock["torch_version"]:
        raise ValueError("Python PyTorch version differs from LibTorch")
    if args.device == "mps" and not python_environment["mps_available"]:
        raise ValueError("MPS unavailable; run with Metal access or select --device cpu")
    envelope = dict(schema="torch.mnist.benchmark.v1", created_at=datetime.now(timezone.utc).isoformat(),
                    config=config, host=dict(platform=platform.platform(), machine=platform.machine(),
                                             processor=processor, cpu_count=os.cpu_count(),
                                             torch_version=lock["torch_version"]),
                    build=stamp, source_sha256=sources, python_environment=python_environment,
                    python_executable_sha256=sha256(python_environment["executable"]),
                    libtorch=lock, dataset_sha256=dataset_hashes,
                    environment={key: env.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                 "PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL")},
                    samples=[], order=[])

    def run_lane(lane, tag, *, prepare=False):
        common = ["--data", args.data.resolve(), "--device", args.device, "--epochs", 1 if prepare else args.epochs,
                  "--batch-size", 32 if prepare else args.batch_size, "--seed", args.seed,
                  "--checkpoint", output / f"{tag}-{lane}.pt"]
        for key in ("train_limit", "test_limit"):
            value = 64 if prepare else getattr(args, key)
            if value is not None:
                common += ["--" + key.replace("_", "-"), value]
        if args.profile and not prepare:
            common += ["--profile"]
        sputnik_options = ["--execution-scope", args.execution_scope]
        commands = {"sputnik-vm": [sputnik, "run", manifest, "--grant", "ffi", "--grant", "fs.read", "--", *common, *sputnik_options],
                    "sputnik-built": [native, *common, *sputnik_options],
                    "python": [python, ROOT / "bench/mnist/python_mnist.py", *common,
                               "--optimizer", args.python_optimizer]}
        stdout, wall = command_result(commands[lane], env, args.timeout, output / f"{tag}-{lane}.log")
        sample = parse_sample(stdout, wall)
        sample.update(lane=lane, command=list(map(str, commands[lane])))
        return sample

    # Executes the FFI loader, cached bytecode graph and model before timing even with --warmups 0.
    print("Preparing VM caches and checking all three implementations…", flush=True)
    prep = [run_lane(lane, "prepare", prepare=True) for lane in LANES]
    prep_args = argparse.Namespace(**vars(args))
    prep_args.train_limit = prep_args.test_limit = 64
    prep_args.batch_size, prep_args.epochs, prep_args.profile = 32, 1, False
    validate_samples(prep, prep_args)
    envelope["preparation"] = prep
    envelope["manifest_sha256"] = sha256(manifest)
    envelope["libtorch_library_sha256"] = {
        name: sha256(Path(lock["torch_prefix"]) / "lib" / name)
        for name in ("libtorch.dylib", "libtorch_cpu.dylib", "libc10.dylib")
        if (Path(lock["torch_prefix"]) / "lib" / name).exists()}
    rng = random.Random(args.order_seed)
    base_order = list(LANES)
    rng.shuffle(base_order)
    for index in range(args.warmups + args.repeats):
        warmup = index < args.warmups
        rotation = index % len(LANES)
        order = base_order[rotation:] + base_order[:rotation]
        tag = f"{'warmup' if warmup else 'repeat'}-{index if warmup else index - args.warmups}"
        envelope["order"].append(dict(tag=tag, lanes=order))
        round_samples = []
        for lane in order:
            print(f"{tag}: {lane}", flush=True)
            sample = run_lane(lane, tag)
            sample.update(warmup=warmup, repeat=index if warmup else index - args.warmups)
            envelope["samples"].append(sample)
            round_samples.append(sample)
            (output / "raw.json").write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n")
            print(f"  loop {sample['loop_seconds']:.3f}s; wall {sample['wall_seconds']:.3f}s", flush=True)
        validate_samples(round_samples, args)
    envelope["summary"] = summarize(envelope["samples"])
    measured = [s for s in envelope["samples"] if not s["warmup"]]
    reference = {s["repeat"]: s for s in measured if s["lane"] == "python"}
    envelope["paired_time_ratios_to_python"] = {
        lane: {phase: stats([s[phase] / reference[s["repeat"]][phase]
                            for s in measured if s["lane"] == lane])
               for phase in ("loop_seconds", "train_seconds", "eval_seconds", "wall_seconds")}
        for lane in LANES if lane != "python"}
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        if baseline["dataset_sha256"] != dataset_hashes:
            raise ValueError("Baseline uses different MNIST files")
        envelope["baseline_comparison"] = compare_baseline(envelope, baseline, args.regression_threshold)
    if not args.profile:
        envelope["python_speed_check"] = check_python_speed(envelope["samples"], args.minimum_speedup)
    (output / "results.json").write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(report(envelope))
    print(report(envelope), flush=True)
    print(f"Saved: {output / 'report.md'}", flush=True)
    if args.require_faster_than_python and not envelope["python_speed_check"]["passed"]:
        return 2
    if args.fail_on_regression and any(c["regression"] for c in envelope["baseline_comparison"]):
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f"MNIST benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
