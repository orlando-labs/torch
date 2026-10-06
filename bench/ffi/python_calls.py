#!/usr/bin/env python3
"""Optional Python controls. Custom Sputnik ABI controls have no Python analogue."""
import argparse
import json
import time

import torch


def constant():
    return 42


def identity(value):
    return value


def identity4(first, second, third, fourth):
    return first


class Identity:
    def constant(self):
        return 42

    def identity(self, value):
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--iterations", type=int, default=300000)
    parser.add_argument("--kernel-iterations", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--case", default="all")
    parser.add_argument("--size", type=int, default=25088)
    args = parser.parse_args()
    if min(args.iterations, args.kernel_iterations, args.repeats, args.size) <= 0:
        parser.error("counts and size must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; CPU fallback is forbidden")
    x = torch.ones(args.size, device=args.device)
    receiver = Identity()

    def measure(name, fn, *, kernel=False):
        if args.case != "all" and args.case != name:
            return
        count = args.kernel_iterations if kernel else args.iterations
        rows = []
        value = None
        for repeat in range(args.repeats + 1):
            if kernel and args.device == "mps":
                torch.mps.synchronize()
            started = time.perf_counter()
            for _ in range(count):
                value = fn()
            if kernel and args.device == "mps":
                torch.mps.synchronize()
            seconds = time.perf_counter() - started
            if repeat:
                rows.append(seconds)
        checksum = value.sum().item() if isinstance(value, torch.Tensor) else value
        print("FFI_ATOM " + json.dumps(dict(name=name, iterations=count,
              seconds=rows, checksum=checksum)), flush=True)

    # Use a lambda in every row: prebinding x.numel would remove the method
    # lookup charged to Sputnik inside its matching captured block.
    measure("loop_scalar", lambda: 42)
    measure("loop_handle", lambda: x)
    measure("sputnik_constant", lambda: constant())
    measure("sputnik_identity_int", lambda: identity(42))
    measure("sputnik_identity4_int", lambda: identity4(42, 1, 2, 3))
    measure("sputnik_identity_handle", lambda: identity(x))
    measure("sputnik_method_constant", lambda: receiver.constant())
    measure("sputnik_method_identity", lambda: receiver.identity(42))
    measure("core_numel_method", lambda: x.numel())
    measure("tensor_numel", lambda: x.numel())
    measure("execution_mode", lambda: 0 if torch.is_grad_enabled() else 1)
    measure("relu_specialized_core", lambda: torch.relu(x), kernel=True)
    measure("relu_generic_core", lambda: torch.relu(x), kernel=True)
    measure("relu_generic_tensor", lambda: torch.relu(x), kernel=True)
    measure("relu_public", lambda: x.relu(), kernel=True)


if __name__ == "__main__":
    main()
