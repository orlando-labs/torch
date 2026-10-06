#!/usr/bin/env python3
"""macOS stack sampling of one long call case, separate from timing runs."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from run import ROOT, sha256, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("sputnik-vm", "sputnik-built"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seconds", type=int, default=6)
    parser.add_argument("--delay", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.iterations, args.seconds) <= 0 or args.delay < 0:
        parser.error("positive durations and iterations are required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sputnik = ROOT.parent / "sputnik-lang/build/sputnik"
    native = ROOT / "build/sputnik/sputniktorch.ffi_calls"
    command = ([str(sputnik), "run", str(ROOT / "build/sputniktorch.ffi_calls.build.json"),
                "--grant", "ffi", "--grant", "fs.read", "--"]
               if args.lane == "sputnik-vm" else [str(native)])
    command += ["--case", args.case, "--iterations", str(args.iterations),
                "--kernel-iterations", str(args.iterations), "--repeats", "1"]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTORCH_ENABLE_MPS_FALLBACK="0")
    for key in ("AMT_BOUNDARY_PROFILE", "SPUTNIK_HEAP_STATS", "DYLD_PRINT_LIBRARIES"):
        env.pop(key, None)
    metadata = dict(command=command, source_sha256=sources(),
                    executable_sha256=sha256(command[0]), iterations=args.iterations,
                    sample_seconds=args.seconds, delay=args.delay)
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            time.sleep(args.delay)
            sampled = subprocess.run(["/usr/bin/sample", str(process.pid), str(args.seconds), "1",
                       "-file", str(output / f"{args.lane}.sample.txt")], capture_output=True, text=True)
            (output / "sample.log").write_text(sampled.stdout + sampled.stderr)
            if sampled.returncode:
                raise RuntimeError(sampled.stdout + sampled.stderr)
            if process.wait(timeout=180):
                raise RuntimeError((output / "run.log").read_text())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    print(output)


if __name__ == "__main__":
    main()
