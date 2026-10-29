# ROCm dependent rebuild — 2026-10-29

Language: `3eb0f05da50ba307d5f719e0b194de6317304b29`. Linux x86_64,
LibTorch 2.14.0, ROCm 6.4.3, three gfx906 AMD devices.

The Linux C++ provider selects torch_hip/c10_hip and propagates USE_ROCM to
preserve the C++ hook ABI. Source adapter builds, native C++ probes and actual
Sputnik VM/full-native checks passed on all three devices. Native graphs have
full coverage, zero VM fallback and are VM independent. Native execution with
PATH=/nonexistent did not load libpython or libtorch_python.

The stock ROCm 6.4.3 rocBLAS package lacks gfx906 Tensile kernels. A signed
AMD ROCm 6.3.4 RPM was extracted into a separate user directory; its gfx906
kernel data is selected through ROCBLAS_TENSILE_LIBPATH. System ROCm shared
libraries remain installed. The origin and SHA-256 are in rocm-result.json.

The local prebuilt package passed installation and VM execution with its
compiler disabled, plus a full-native installed consumer. The production
notebook worker installed that package through the API and passed GPU matrix
multiplication, gradients and SGD training. Receipt replay preserved RunId;
the test project was deleted. The same package was installed in the user's
normal package home with a runnable example.

macOS CPU checks passed in VM/full-native modes. Actual MPS checks passed
with CPU fallback disabled. ROCm capability predicates return Bool/false on
macOS. Notebook snapshot and C++ provider selection tests passed.

The package is local, keeps LibTorch/Tensile external and is not relocatable
or a public registry artifact. See docs/package-building.md for the commands.
