#!/usr/bin/env python3
"""Sequential FFI decomposition with provenance and checked results.

Build via tools/build first; this optional Python runner does not build the
package. No builds, tests or other benchmarks may run during measurement.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "FFI_ATOM "
LANES = ("sputnik-vm", "sputnik-built", "python")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources():
    paths = [ROOT / "cmake/BuildSputnik.cmake", ROOT / "tools/build"]
    for directory, patterns in (("src", ("*.s",)), ("native", ("*.cpp", "*.h")),
                                ("bench/ffi", ("*.s", "*.cpp", "*.h", "*.py"))):
        for pattern in patterns:
            paths.extend((ROOT / directory).rglob(pattern))
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(paths))}


def validate(rows, args, lane):
    expected = dict(loop_scalar=42, loop_handle=args.size, sputnik_constant=42,
                    sputnik_identity_int=42, sputnik_identity_handle=args.size,
                    sputnik_identity4_int=42, abi_identity4_int=42,
                    sputnik_method_constant=42, sputnik_method_identity=42,
                    abi_constant=42, abi_identity_int=42, abi_identity_handle=args.size,
                    abi_int_rebox=42, abi_handle_checked=args.size,
                    abi_numel_free=args.size, core_numel_method=args.size,
                    tensor_numel=args.size, execution_mode=0, abi_string_4=4,
                    abi_string_256=256, abi_list_len_4=4, abi_list_sum_4=10,
                    abi_list_sum_32=496, relu_specialized_core=args.size,
                    relu_generic_core=args.size, relu_generic_tensor=args.size,
                    relu_public=args.size, cpp_constant=42, cpp_identity_int=42,
                    cpp_numel=args.size, cpp_relu=args.size)
    if args.case != "all" and args.case not in expected:
        raise ValueError(f"Unknown benchmark case: {args.case}")
    if lane == "python":
        expected = {key: value for key, value in expected.items()
                    if not key.startswith(("abi_", "cpp_"))}
    if args.case != "all":
        expected = {key: value for key, value in expected.items() if key == args.case}
    if {row["name"] for row in rows} != set(expected) or len(rows) != len(expected):
        raise ValueError(f"{lane}: missing, extra or duplicate benchmark cases")
    for row in rows:
        iterations = args.kernel_iterations if "relu" in row["name"] else args.iterations
        if (row["iterations"] != iterations or row["checksum"] != expected[row["name"]]
                or len(row["seconds"]) != args.repeats or
                any(not math.isfinite(t) or t <= 0 for t in row["seconds"])):
            raise ValueError(f"{lane}: invalid result: {row}")


def summarize(samples):
    result = {}
    for sample in samples:
        lane = result.setdefault(sample["lane"], {})
        for row in sample["rows"]:
            lane.setdefault(row["name"], []).append(
                statistics.median(row["seconds"]) * 1e9 / row["iterations"])
    return {lane: {key: dict(median_ns=statistics.median(values),
                             min_ns=min(values), max_ns=max(values), processes_ns=values)
                   for key, values in cases.items()} for lane, cases in result.items()}


def report(envelope):
    summary = envelope["summary"]
    lanes = envelope.get("lanes", LANES)
    lines = [f"# FFI calls — {envelope['config']['device']} — size {envelope['config']['size']}", "",
             "Nanoseconds per iteration; median of each process's medians. "
             "Ordinary rows include a shared loop/block harness. cpp_* rows use a C++ loop.", "",
             "Custom ABI controls have no Python counterpart; — is intentional.", "",
             "| Case | " + " | ".join(f"{lane}, ns/op" for lane in lanes) + " |",
             "| --- | " + " | ".join("---:" for _ in lanes) + " |"]
    for key in summary["sputnik-vm"]:
        cells = []
        for lane in lanes:
            value = summary[lane].get(key)
            cells.append(f"{value['median_ns']:.2f}" if value else "—")
        lines.append("| " + key + " | " + " | ".join(cells) + " |")
    contrasts = (("Sputnik free call", "sputnik_constant", "loop_scalar"),
                 ("Sputnik method call", "sputnik_method_constant", "loop_scalar"),
                 ("ABI identity + one argument", "abi_identity_int", "loop_scalar"),
                 ("ABI call versus Sputnik free call", "abi_constant", "sputnik_constant"),
                 ("Three additional scalar ABI arguments", "abi_identity4_int", "abi_identity_int"),
                 ("Int decoding/reboxing + boundary", "abi_int_rebox", "abi_identity_int"),
                 ("Handle tag validation + boundary", "abi_handle_checked", "abi_identity_handle"),
                 ("Read numel + Int result", "abi_numel_free", "abi_handle_checked"),
                 ("List[4] conversion", "abi_list_sum_4", "abi_list_len_4"),
                 ("Additional 28 list elements", "abi_list_sum_32", "abi_list_sum_4"),
                 ("Generic operation/options dispatch", "relu_generic_core", "relu_specialized_core"),
                 ("Sputnik Tensor wrapper", "relu_generic_tensor", "relu_generic_core"),
                 ("Public wrapper / execution mode", "relu_public", "relu_generic_tensor"))
    sputnik_lanes = [lane for lane in lanes if lane != "python"]
    lines += ["", "Paired differences: median of within-process row-median differences. "
              "These are controlled contrasts, not additive or pure ABI timings.", "",
              "| Contrast | " + " | ".join(f"{lane}, ns" for lane in sputnik_lanes) + " |",
              "| --- | " + " | ".join("---:" for _ in sputnik_lanes) + " |"]
    for label, high, low in contrasts:
        cells = []
        for lane in sputnik_lanes:
            if high not in summary[lane] or low not in summary[lane]:
                cells.append("—")
            else:
                values = [a - b for a, b in zip(summary[lane][high]["processes_ns"],
                                               summary[lane][low]["processes_ns"])]
                cells.append(f"{statistics.median(values):.2f}")
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--size", type=int, default=25088)
    parser.add_argument("--iterations", type=int, default=300000)
    parser.add_argument("--kernel-iterations", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--case", default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--python", default="/opt/homebrew/opt/pytorch/libexec/bin/python")
    parser.add_argument("--vm-sputnik", type=Path,
                        help="VM binary for before/after comparison; native build still validates its own compiler")
    parser.add_argument("--before-vm", type=Path,
                        help="Add a saved VM to the same rotating process order for a paired A/B")
    parser.add_argument("--build-log", type=Path, default=ROOT / "build/ffi-calls-build.log")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.size, args.iterations, args.kernel_iterations, args.repeats,
           args.processes, args.threads) <= 0:
        parser.error("counts and size must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    native_build_sputnik = ROOT.parent / "sputnik-lang/build/sputnik"
    sputnik = args.vm_sputnik.resolve() if args.vm_sputnik else native_build_sputnik
    manifest = ROOT / "build/sputniktorch.ffi_calls.build.json"
    native = ROOT / "build/sputnik/sputniktorch.ffi_calls"
    provenance = sources()
    executable_hashes = {"sputnik": sha256(sputnik), "native": sha256(native),
                         "python": sha256(args.python),
                         "native_build_sputnik": sha256(native_build_sputnik)}
    lanes = list(LANES)
    if args.before_vm:
        args.before_vm = args.before_vm.resolve()
        executable_hashes["before_vm"] = sha256(args.before_vm)
        lanes.insert(1, "vm-before")
    build_inputs = json.loads((ROOT / "build/ffi-calls-build-inputs.json").read_text())
    if build_inputs["native_sha256"] != executable_hashes["native"]:
        raise ValueError("Binary changed since input fingerprint; rebuild calls.s")
    for path, expected in build_inputs["inputs"].items():
        if sha256(path) != expected:
            raise ValueError(f"Input changed since native build: {path}")
    build_text = args.build_log.read_text()
    build, _ = json.JSONDecoder().raw_decode(build_text[build_text.index("{"):])
    if (not build["native_graph_full_coverage"] or build["native_bytecode_fallback"]
            or build["native_hash"] != executable_hashes["native"]):
        raise ValueError("Stale binary or incomplete native coverage; rebuild calls.s")
    for artifact in build["artifacts"]:
        if sha256(artifact["path"]) != artifact["source_hash"]:
            raise ValueError(f"Source changed since native build: {artifact['path']}")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               PYTORCH_ENABLE_MPS_FALLBACK="0",
               SPUTNIK_NATIVE_RUNTIME_ROOT=str(ROOT.parent / "sputnik-lang"))
    for key in ("AMT_BOUNDARY_PROFILE", "SPUTNIK_HEAP_STATS", "DYLD_PRINT_LIBRARIES"):
        env.pop(key, None)
    common = ["--device", args.device, "--size", str(args.size),
              "--iterations", str(args.iterations), "--kernel-iterations", str(args.kernel_iterations),
              "--repeats", str(args.repeats), "--case", args.case]
    commands = {"sputnik-vm": [str(sputnik), "run", str(manifest), "--grant", "ffi", "--grant", "fs.read", "--"],
                "sputnik-built": [str(native)],
                "python": [args.python, str(ROOT / "bench/ffi/python_calls.py")]}
    if args.before_vm:
        commands["vm-before"] = [str(args.before_vm), "run", str(manifest),
                                 "--grant", "ffi", "--grant", "fs.read", "--"]
    envelope = dict(schema="sputniktorch.ffi.calls.v1", created_at=datetime.now(timezone.utc).isoformat(),
                    config={key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
                    platform=platform.platform(), source_sha256=provenance,
                    manifest_sha256=sha256(manifest), executables_sha256=executable_hashes,
                    libtorch=json.loads((ROOT / "build/native-lock.json").read_text()),
                    lanes=lanes, samples=[])
    envelope["build_inputs"] = build_inputs
    envelope["native_build"] = {key: build[key] for key in (
        "native_hash", "native_backend", "native_graph_full_coverage",
        "native_bytecode_fallback", "native_extensions")}
    envelope["libtorch_library_sha256"] = {
        name: sha256(Path(envelope["libtorch"]["torch_prefix"]) / "lib" / name)
        for name in ("libtorch.dylib", "libtorch_cpu.dylib", "libc10.dylib")}
    py_info = subprocess.check_output([args.python, "-c", "import json,sys,torch; "
        "print(json.dumps(dict(python=sys.version, torch=torch.__version__, "
        "torch_path=torch.__file__, mps=torch.backends.mps.is_available())))"], env=env, text=True)
    envelope["python_environment"] = json.loads(py_info)
    if envelope["python_environment"]["torch"] != envelope["libtorch"]["torch_version"]:
        raise ValueError("PyTorch and LibTorch versions differ")
    if args.device == "mps" and not envelope["python_environment"]["mps"]:
        raise RuntimeError("MPS unavailable; run with Metal access")
    # Cache preparation and correctness checks are excluded from measurements.
    for lane in lanes:
        command = commands[lane] + common
        prep = command.copy()
        for option in ("--iterations", "--kernel-iterations", "--repeats"):
            prep[prep.index(option) + 1] = "2"
        print("prepare:", lane, flush=True)
        result = subprocess.run(prep, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        (output / f"prepare-{lane}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{lane} preparation failed: {result.stderr[-1500:]}")
        prep_args = argparse.Namespace(**vars(args))
        prep_args.iterations = prep_args.kernel_iterations = prep_args.repeats = 2
        validate([json.loads(line[len(PREFIX):]) for line in result.stdout.splitlines()
                  if line.startswith(PREFIX)], prep_args, lane)
    for repeat in range(args.processes):
        rotation = repeat % len(lanes)
        for lane in lanes[rotation:] + lanes[:rotation]:
            print(f"process {repeat}:", lane, flush=True)
            command = commands[lane] + common
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
            (output / f"{repeat}-{lane}.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"{lane} failed: {result.stderr[-1500:]}")
            rows = [json.loads(line[len(PREFIX):]) for line in result.stdout.splitlines()
                    if line.startswith(PREFIX)]
            validate(rows, args, lane)
            envelope["samples"].append(dict(lane=lane, repeat=repeat, command=command, rows=rows))
            (output / "raw.json").write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n")
    final_hashes = {
            "sputnik": sha256(sputnik), "native": sha256(native), "python": sha256(args.python),
            "native_build_sputnik": sha256(native_build_sputnik)}
    if args.before_vm:
        final_hashes["before_vm"] = sha256(args.before_vm)
    if provenance != sources() or executable_hashes != final_hashes:
        raise RuntimeError("Sources or executables changed during measurement")
    envelope["summary"] = summarize(envelope["samples"])
    (output / "results.json").write_text(json.dumps(envelope, indent=2, allow_nan=False) + "\n")
    rendered = report(envelope)
    (output / "report.md").write_text(rendered)
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
