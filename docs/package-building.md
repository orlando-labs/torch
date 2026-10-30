# Building distribution packages

The v1 builder supports three independent axes: Sputnik source/bytecode/mixed,
native source/prebuilt, and third-party dependency bundle none/runtime/sdk.
The default builds one source/source/none artifact without probing LibTorch,
Homebrew, a C++ compiler or a GPU:

```sh
make -C ../sputnik-lang build/sputnik build/package_distribution_tests
tools/package
# A particular artifact:
tools/package --native source --modules bytecode --bundle none
# All 15 supported current-platform combinations plus separate providers:
tools/build --fetch-deps
tools/package --matrix --out-dir build/packages-v1-bundles
tools/test-packages build/packages-v1-bundles
```

| Native adapter | Bundle | Third-party inputs at consumer preparation |
|---|---|---|
| source | none | Matching runtime and SDK providers; C++ compiler |
| source | runtime | Matching SDK provider; C++ compiler |
| source | sdk | Host C++ compiler and platform SDK |
| prebuilt | none | Runtime provider |
| prebuilt | runtime | No SDK or C++ compiler for VM execution |

Prebuilt+sdk is rejected. Mixed keeps the root Sputnik facade as source and
compiles other eligible modules to bytecode. Bytecode does not hide C++ source
when the native adapter axis is source. Current recipes target macOS arm64/x86_64;
CPU and actual MPS share one runtime closure. The verified snapshot uses LibTorch
2.14.0 and macOS aarch64, minimum macOS 26.0 (the binaries' actual minimum).
Source/none remains conservatively platform-specific because its SDK requirements
have only been verified on this target.

The additional options are `--torch-prefix`, `--sputnik-root`, `--version`,
`--out-dir` and `--provider-spec`. Defaults are the neighboring Sputnik toolchain,
version `0.1.0-dev` and `build/packages`. Only variants needing dependency
preparation resolve Homebrew's PyTorch prefix. `--provider-spec` takes the small
CMake metadata file produced by a matrix build; it pins the build identity for a
reproducible source-only repack without installing the dependencies.

No Python dependency is used. Native/bundle preparation requires CMake, Ninja,
LibTorch and pinned Gloo headers prepared by `tools/build`; macOS relocation uses
`otool`, `lipo`, `install_name_tool` and `codesign`. CPU/MPS probe execution is
required for this build recipe. A process without Metal access cannot satisfy MPS
verification.

The matrix produces two additional native-only `.sputnikpkg` files:
`torch-libtorch-runtime` and `torch-libtorch-sdk`. They have providers,
inventories and license materials, with no dummy Sputnik modules. Extracted
provider directories are explicit local bindings:

```sh
export SPUTNIK_NATIVE_PROVIDERS=/absolute/providers/runtime:/absolute/providers/sdk
export SPUTNIK_NATIVE_EXT_CACHE=/absolute/writable/native-cache
sputnik run consumer.s --grant ffi
```

Use both bindings for source/none, only SDK for source/runtime, only runtime for
prebuilt/none, and neither for source/sdk or prebuilt/runtime. Requirements pin
version/ABI and, for the matrix, exact producer build identity. SDK/runtime builds
must match. The loader checks provider inventories and creates canonical cached
provider roots; prebuilt adapters use an immutable copied library plus a local
`lib/vendor` directory binding. Installation does not patch or resign binaries.
Provider fingerprints participate in the source native compilation cache.
Conflicting singleton providers are rejected before loading.

Runtime bundles contain the declared import closure, not the whole PyTorch
prefix. Source/sdk adds compiler-observed headers plus pinned Gloo; this is the
SDK needed by this adapter/target, not a universal developer distribution.
Third-party component mappings and retained LICENSE/NOTICE/exception/formula
materials appear in `meta/components.json` and `licenses/`. Recipe validation
rejects missing attribution mappings and materials; upstream embedded/header
terms remain in their original files and notices. This metadata is not a legal
clearance for publication.

`report.json` records all artifacts, separate providers, targets, inventories,
sizes and digests. `REPORT.md` is the readable projection. Verification extracts
all 15 into a relocated consumer tree, runs actual CPU/MPS forward/backward with
MPS fallback disabled, runs prebuilt VM tests with an unavailable C++ compiler,
and builds one bytecode consumer for each of the five native/bundle combinations
with `--require-full-native`. Prepared executables run without SDK bindings.
A slim repack with a nonexistent dependency prefix must reproduce its matrix
artifact. `verification/result.json` binds results to the exact report digest.
All Sputnik invocations are sequential.

The macOS matrix above produces local development artifacts. The earlier six bundles in
`build/packages` remain unchanged; the expanded matrix is in
`build/packages-v1-bundles`. Shared installation/registry variant selection,
managed provider dependency resolution and portable application locks are not
implemented by that matrix recipe itself. The shared installer now supports
registry selection, exact technical dependencies and variant/hash locks; see
[native installation](../../sputnik-lang/docs/engineering/package-native-install-v1.md).
Extracted local overrides and explicit provider bindings remain supported.

## Local Linux ROCm builds

For a local Linux ROCm C++ provider, `tools/build --torch-prefix DIR` detects
`libtorch_hip.so` and `libc10_hip.so`, links those libraries explicitly and uses
the selected `SPUTNIK_NATIVE_CXX`/`CXX` compiler. The adapter does not compile HIP
kernels or invoke Python, nvcc or hipcc. The provider must supply LibTorch
2.14.0 or 2.14.1 headers and shared libraries; runtime ROCm libraries remain
dependencies of that provider.

Both the adapter and generated Sputnik consumers compile with `USE_ROCM` to
match LibTorch's ROCm C++ hook ABI. Runtime dependencies follow the provider's
recorded library paths. When several RCCL versions are installed, prepending
an older ROCm library directory can override the provider's matching RCCL.

ROCm tensors use `device: "cuda:0"`, following
[PyTorch HIP semantics](https://docs.pytorch.org/docs/stable/notes/hip.html).
`torch.backends.rocm_built?` identifies a HIP build and `rocm_available?` also
requires a visible GPU. The existing CUDA availability, count, synchronization
and tensor APIs work on HIP builds. This local build path does not produce or
publish a CUDA registry artifact.

```sh
tools/build tests/rocm.s --torch-prefix /path/to/libtorch --run-vm --no-notebook
tools/build tests/rocm.s --torch-prefix /path/to/libtorch --no-notebook
build/sputnik/torch.rocm_test
```

The ROCm test runs tensor placement, forward, backward, a training update,
byte transfer, seeded generation and synchronization on the selected AMD GPU.
Set `HIP_VISIBLE_DEVICES` to select a physical GPU; each invocation addresses
the visible device as `cuda:0`.

`tools/test-rocm --torch-prefix DIR --devices '0 1 2'` builds with full native
coverage and tests each visible AMD device in VM and native modes. Native
execution runs with an unavailable executable search path and rejects loaded
`libpython` or `libtorch_python` libraries.

An adapter already built from these sources can be tested through the package
ABI with `--native-library build/rocm-package-native/libtorch_notebook.so`.
The test copies it into a content-addressed directory, runs the Sputnik graph
in VM mode and builds that graph with `--require-full-native`.

On gfx906 hosts, check that rocBLAS ships the corresponding Tensile kernels.
ROCm 6.4 packages can omit them. A separate matching kernel data directory can
be selected with `ROCBLAS_TENSILE_LIBPATH`; the local package records that path
alongside the C++ provider. Preserve the same setting when running its native
consumers or a notebook worker.

For installation on the same server, a local prebuilt adapter package can be
created and checked through an isolated package home:

```sh
tools/package --backend rocm --native prebuilt --modules source --bundle none \
  --torch-prefix /path/to/libtorch --version 0.1.3-rocm-local \
  --out-dir build/rocm-local-package
```

This package records the host provider path and hash, carries the Sputnik
modules and adapter, and keeps LibTorch/ROCm external. It is not relocatable
and is not a public registry artifact. Its preparation verifies an installed
consumer in VM with the compiler disabled and in full native mode, including
forward, backward and an SGD training update on AMD.

## Linux ROCm registry package

The runtime recipe carries LibTorch 2.14.0 and matching RCCL 2.27.7, with
relative library paths and scoped modules. ROCm 6.4.3 userspace, the AMD driver
and rocBLAS Tensile data remain system dependencies. The recipe verifies the
completed archive on three gfx906 GPUs, in VM and full native modes, without
producer library paths. It does not invoke Python.

```sh
tools/package --backend rocm --native prebuilt --modules source --bundle runtime \
  --package-ref slowpilot/torch --version 0.1.3 \
  --torch-prefix /path/to/libtorch --out-dir build/registry-013-rocm
sputnik pkg install slowpilot/torch --version 0.1.3 \
  --backend rocm --bundle runtime \
  --allow-native --allow-native-large
```

The package manager selects this variant with `--backend rocm` and verifies a
visible GPU through the installed HIP C runtime. CPU installs do not select a
ROCm-only variant. The native provider records the exact ROCm LibTorch ABI and
content hash. On gfx906, retain `ROCBLAS_TENSILE_LIBPATH` if the system
rocBLAS installation lacks its kernel data. The package uses Sputnik's current
bare function semantics; callback references use `&name`.

## Linux CUDA runtime package

The CUDA recipe accepts exactly **LibTorch 2.14.0+cu130** and uses its C++
distribution, without Python or a host CUDA compiler dependency. It builds the
adapter with `-O3`, runs CPU and each selected GPU forward/backward probes, and
bundles runtime libraries, hashes, requirements and upstream license materials.
GPU identifiers are resolved to UUIDs before running CUDA to avoid confusing
NVML indices with CUDA ordinals.

```sh
tools/package --backend cuda --native prebuilt --modules source --bundle runtime \
  --package-ref slowpilot/torch --version 0.1.2 --cuda-devices '1;2' \
  --torch-prefix /absolute/libtorch-2.14.0-cu130 --sputnik-root ../sputnik-lang \
  --out-dir /absolute/packages
```

This produces `linux-x86_64-gnu-cu130-prebuilt-source-runtime`. The manifest
requires exact LibTorch `2.14.0`, the C++11/CUDA ABI and a SHA-256 build identity
derived from every bundled library. A different LibTorch version does not satisfy
it. Only compute capabilities actually tested by the recipe are advertised.
On the host reached through SSH port 22176 those are RTX 3090 (8.6) and RTX 4090 (8.9); its GTX 1080 Ti
(6.1) is outside this CUDA 13 artifact's declared targets. Use
`CUDA_VISIBLE_DEVICES` with a supported GPU UUID when installing/running there.

`tests/registry_cuda_check.s` verifies real CUDA placement, forward and backward
through the scoped published modules. `tests/registry_cuda_remote.s` uploads it
with an exact lock to a real notebook remote core, installs the registry package
through the dependency API and verifies the persisted Run result.

The recipe retains the adapter's ELF SONAME as its archive filename. Before
publication, it installs the completed archive in a clean store, builds with
`--require-full-native`, and runs both VM and native checks on each selected
GPU without the producer's `LD_LIBRARY_PATH`. All Sputnik invocations are
sequential; no Python is required.

See the [generic distribution API](../../sputnik-lang/docs/engineering/package-distribution-v1.md)
and [native dependency design](../../sputnik-registry/docs/NATIVE_BUNDLES.md).

## Scoped macOS release variants

macOS packages accept `--package-ref slowpilot/torch`, preserving the
same scoped module declarations and error classes as the CUDA release. For a
prebuilt CPU/MPS runtime package and a thin native-source package:

```sh
tools/package --package-ref slowpilot/torch --version 0.1.2 \
  --native prebuilt --modules source --bundle runtime \
  --out-dir build/registry-012-macos-20261022
tools/package --package-ref slowpilot/torch --version 0.1.2 \
  --native source --modules source --bundle none \
  --provider-spec build/registry-012-macos-20261022/provider-spec.cmake \
  --out-dir build/registry-012-source-20261022
```

The prebuilt runtime needs no consumer C++ compiler or SDK. The thin source
archive contains Sputnik/C++ sources and headers belonging to this adapter;
LibTorch runtime/SDK and Gloo are external. Its exact LibTorch 2.14.0, ABI and
SDK/runtime build requirements are checked before compilation. Use prepared
providers from the matching local dependency build:

```sh
export SPUTNIK_NATIVE_PROVIDERS="$(pwd)/build/registry-012-macos-20261022/providers/runtime:$(pwd)/build/registry-012-macos-20261022/providers/sdk"
sputnik pkg install torch --version 0.1.2 \
  --variant macos-aarch64-source-source-none --allow-native --build-from-source
```

These two variants are verified for macOS aarch64 (minimum macOS 26.0 for this
dependency build), with actual CPU/MPS execution and MPS fallback disabled.
The thin source target retains those platform restrictions; it does not claim
unverified Linux/Windows portability. Missing providers and wrong LibTorch
versions must fail before a compiler runs. Neither thin source packaging nor
consumer preparation invokes Python.

`tests/registry_macos_packages.cmake` checks clean-store installation, exact
variant/hash locks, VM and full-native execution of both completed archives,
including the scoped device error, bytes and seeded GPU generation.
