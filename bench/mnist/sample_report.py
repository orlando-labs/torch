#!/usr/bin/env python3
"""Summarize macOS sample trees with exclusive weights and per-thread attribution.

Wait samples on different threads must not be summed into a wall-time percentage.
Inclusive runtime weights identify ancestors; they must not be summed together.
"""

import argparse
import collections
import json
import pathlib
import re

WAIT = re.compile(
    r"(?:__psynch_cvwait|__psynch_mutexwait|__ulock_wait|semaphore_wait|"
    r"semaphore_timedwait|mach_msg2?_trap|kevent|__workq_kernreturn|__semwait_signal)"
)
BRIDGE_FUNCTION = re.compile(r"\b(amt_[A-Za-z0-9_]+)")


def realm(path):
    for frame in reversed(path):
        match = re.search(r"\(in ([^)]+)\)", frame)
        if not match:
            continue
        # Native links the bridge into the executable; VM loads its dylib.
        # Attribute both C ABI functions and inlined bridge helpers equally.
        function = frame.split(" (in ", 1)[0].strip()
        if function.startswith(("amt_", "torch::")):
            return "Torch bridge"
        binary = match[1]
        # The extension also starts with "sputnik". Check its exact name
        # before the executable prefix so its C++ helpers and allocations
        # are not charged to the VM.
        if binary == "sputnik_manifest_extensions.dylib":
            return "Torch bridge"
        if binary.startswith(("sputnik", "torch.mnist_", "torch.ffi_calls")):
            return "Sputnik runtime"
        if binary.startswith(("libtorch", "libc10", "libshm")):
            return "LibTorch"
        if binary.startswith(("AGX", "Metal", "MPS", "IOGPU", "AppleNeural")):
            return "Metal/MPS"
        if binary.startswith(("Python", "_C.cpython")):
            return "Python runtime"
    return "Other"


def parse_tree(text):
    graph = text.split("Call graph:\n", 1)[1].split("\nTotal number in stack", 1)[0]
    roots, stack = [], []
    for line in graph.splitlines():
        match = re.match(r"^([ +!:|]*)(\d+) (.+)$", line)
        if not match:
            continue
        depth, count, frame = len(match[1]), int(match[2]), match[3]
        while stack and stack[-1]["depth"] >= depth:
            stack.pop()
        node = dict(depth=depth, n=count, f=frame, children=[])
        (stack[-1]["children"] if stack else roots).append(node)
        stack.append(node)
    return roots


def summarize(root):
    rows, operations, hot, inclusive = (collections.Counter() for _ in range(4))

    def walk(node, path):
        path = path + [node["f"]]
        exclusive = node["n"] - sum(child["n"] for child in node["children"])
        if exclusive < 0:
            raise ValueError("Child weights exceed their parent: " + node["f"])
        if exclusive:
            state = "wait" if WAIT.search(node["f"]) else "active"
            category = realm(path)
            rows[(state, category)] += exclusive
            operation = next((match[1] for frame in reversed(path)
                              if (match := BRIDGE_FUNCTION.search(frame))), None)
            if operation:
                operations[(state, operation)] += exclusive
            if state == "active" and category == "Sputnik runtime":
                hot[re.sub(r" \(in .*", "", node["f"])] += exclusive
            # Count each ancestor at most once per leaf, including recursion.
            if state == "active":
                names = {re.sub(r" \(in .*", "", frame)
                         for frame in path if realm([frame]) == "Sputnik runtime"}
                for name in names:
                    inclusive[name] += exclusive
        for child in node["children"]:
            walk(child, path)

    walk(root, [])
    if sum(rows.values()) != root["n"]:
        raise ValueError("Exclusive weights do not cover the thread")
    return dict(thread=root["f"], samples=root["n"],
                states={str(key): value for key, value in rows.items()},
                ops={str(key): value for key, value in operations.most_common(16)},
                runtime_hot=hot.most_common(20), runtime_inclusive=inclusive.most_common(30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    args = parser.parse_args()
    for file in sorted(args.directory.glob("*.sample.txt")):
        threads = [summarize(root) for root in parse_tree(file.read_text())]
        file.with_suffix(".summary.json").write_text(json.dumps(threads, indent=2) + "\n")
        print(file.name)
        for thread in threads:
            if sum(value for key, value in thread["states"].items() if "active" in key) < 20:
                continue
            print(thread["thread"], thread["samples"], thread["states"])
            print("ops", thread["ops"])
            print("runtime", thread["runtime_hot"][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
