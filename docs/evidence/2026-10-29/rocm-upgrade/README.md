# ROCm dependent rebuild — 2026-10-29

Target: `slowpilot@lab.orlando-labs.com:22015`, host `dev-ai-amd`.
Read-only inventory found ROCm 6.4.3, three gfx906 AMD devices and a LibTorch
2.14.0 C++ HIP provider under `/usr/local/lib64/python3.12/site-packages/torch`.
The provider's C++ headers/libraries are consumed directly; the package build
and tests never invoke Python or link its runtime.

Added HIP library detection for Linux C++ providers, a compiler selection that
works without clang++, ROCm capability predicates, VM/full-native AMD tests,
a local package producer and a remote-worker package/training test. The local
package keeps the recorded host LibTorch/ROCm external and is not relocatable
or a public registry release.

CPU/CUDA/ROCm provider selection and missing-library checks passed. Notebook
bundle snapshot/registration checks passed. Current macOS torch CPU checks
passed in VM and full native modes. Actual MPS validation passed with automatic CPU fallback disabled, including CUDA-tagged checkpoint transfer and shared storage. Both native graphs have full coverage, zero VM fallback and are VM independent.

Remote source upload and deployment are pending explicit user authorization:
automatic approval review rejected transfer of the current local language and
torch sources plus notebook sources from port 22176 to port 22015. No source
transfer was retried. The new server's build dependencies and a private PCRE2
10.47 prefix were prepared; Sputnik/remote/torch are not installed there yet.
