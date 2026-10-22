#!/usr/bin/env python3
"""Optional MNIST stack samples or matched atomic costs, outside throughput runs.

Build benchmark.s or atoms.s first with tools/build. Diagnostic runs are
sequential; do not run builds, tests or other benchmarks alongside them.
"""

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

from run import ROOT, sha256, source_hashes

PREFIX = "MNIST_ATOM "
BOUNDARY_PREFIX = "AMT_BOUNDARY_PROFILE "


def atom_report(results):
    refs = {row["name"]: row for row in results["python"]["atoms"]}
    if not refs:
        raise ValueError("No Python atomic results")
    lines = ["| Case | before VM, µs/op | VM, µs/op | native, µs/op | Python, µs/op |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, reference in refs.items():
        values = []
        for lane in ("before-vm", "sputnik-vm", "sputnik-built", "python"):
            row = next((row for row in results.get(lane, {}).get("atoms", [])
                        if row["name"] == key), None)
            if row is None:
                if lane != "before-vm":
                    raise ValueError(f"{lane} missing {key}; rebuild atoms")
                values.append("—")
                continue
            if (row["iterations"] != reference["iterations"] or
                    abs(row["checksum"] - reference["checksum"]) >
                    1e-5 * max(1, abs(reference["checksum"]))):
                raise ValueError(f"Atomic numerical mismatch {key} {lane}")
            values.append(f"{statistics.median(row['seconds']) * 1e6 / row['iterations']:.3f}")
        lines.append("| " + key + " | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--atoms", action="store_true")
    parser.add_argument("--boundary", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--sample-seconds", type=int, default=6)
    parser.add_argument("--delay", type=float, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--python", default="/opt/homebrew/opt/pytorch/libexec/bin/python")
    parser.add_argument("--before-vm", type=pathlib.Path)
    parser.add_argument("--data", type=pathlib.Path, default=ROOT / "build/datasets/mnist")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.sample and args.atoms:
        parser.error("sample uses a long MNIST run; atom cases are short")
    if min(args.epochs, args.sample_seconds, args.threads) <= 0 or args.delay < 0:
        parser.error("positive durations, threads and epochs are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               PYTORCH_ENABLE_MPS_FALLBACK="0")
    for key in ("AMT_BOUNDARY_PROFILE", "SPUTNIK_HEAP_STATS", "DYLD_PRINT_LIBRARIES"):
        env.pop(key, None)
    if args.boundary:
        env["AMT_BOUNDARY_PROFILE"] = "1"
    sputnik = ROOT.parent / "sputnik-lang/build/sputnik"
    name = "mnist_atoms" if args.atoms else "mnist_benchmark"
    manifest = ROOT / f"build/torch.{name}.build.json"
    native = ROOT / f"build/sputnik/torch.{name}"
    common = ["--data", str(args.data.resolve()), "--device", args.device]
    if not args.atoms:
        common += ["--epochs", str(args.epochs), "--batch-size", "32",
                   "--train-limit", "2048", "--test-limit", "512"]
    commands = {}
    if args.before_vm:
        commands["before-vm"] = [str(args.before_vm.resolve()), "run", str(manifest),
                                 "--grant", "ffi", "--grant", "fs.read", "--"]
    commands["sputnik-vm"] = [str(sputnik), "run", str(manifest), "--grant", "ffi",
                            "--grant", "fs.read", "--"]
    commands["sputnik-built"] = [str(native)]
    python_file = "python_atoms.py" if args.atoms else "python_mnist.py"
    commands["python"] = [args.python, str(ROOT / "bench/mnist" / python_file)]
    metadata = dict(created_at=datetime.now(timezone.utc).isoformat(), device=args.device,
                    threads=args.threads, atoms=args.atoms, boundary=args.boundary,
                    sampled=args.sample, sample_seconds=args.sample_seconds, delay=args.delay,
                    source_sha256=source_hashes(), manifest_sha256=sha256(manifest),
                    executables={lane: sha256(command[0]) for lane, command in commands.items()},
                    results={})
    for lane, prefix in commands.items():
        command = prefix + common
        if not args.atoms:
            command += ["--checkpoint", str(output / f"{lane}.pt")]
            if lane == "python":
                command += ["--optimizer", "packed"]
        print(lane, flush=True)
        log = output / f"{lane}.log"
        started = time.perf_counter()
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT)
            try:
                if args.sample:
                    time.sleep(args.delay)
                    subprocess.run(["/usr/bin/sample", str(process.pid), str(args.sample_seconds),
                                    "1", "-file", str(output / f"{lane}.sample.txt")],
                                   check=True, capture_output=True)
                code = process.wait(timeout=600)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        if code:
            raise RuntimeError(f"{lane} exit {code}: {log.read_text()[-2000:]}")
        text = log.read_text()
        records = [json.loads(line[len(PREFIX):]) for line in text.splitlines()
                   if line.startswith(PREFIX)]
        boundary = [json.loads(line[len(BOUNDARY_PREFIX):]) for line in text.splitlines()
                    if line.startswith(BOUNDARY_PREFIX)]
        metadata["results"][lane] = dict(command=command, wall_seconds=time.perf_counter() - started,
                                         atoms=records, boundary=boundary)
        (output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.atoms:
        report = atom_report(metadata["results"])
        (output / "report.md").write_text(report)
        print(report)
    if args.sample:
        with (output / "analysis.log").open("w") as stream:
            subprocess.run([sys.executable, str(ROOT / "bench/mnist/sample_report.py"), str(output)],
                           check=True, stdout=stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
