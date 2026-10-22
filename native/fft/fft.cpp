#include "tensor/kernel_args.h"

using namespace sputnik_torch;
extern "C" SputnikStatus amt_fft(SputnikCtx *cx, const SputnikValue *a, size_t n,
                                  SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]));
    KernelArgs args(cx, a[1], a[2]);
    const auto op = string(cx, a[0]);
    const auto x = args.tensor(0);
    at::Tensor r;
    if (op == "fftshift" || op == "ifftshift") {
      const auto dim_values = args.optional_integers(0);
      const auto dim = optional_dims(dim_values);
      r = op == "fftshift" ? at::fft_fftshift(x, dim) : at::fft_ifftshift(x, dim);
    } else {
      const auto norm = args.string(2);
      if (op == "fft") r = at::fft_fft(x, args.optional_integer(0), args.integer(1), norm);
      else if (op == "ifft") r = at::fft_ifft(x, args.optional_integer(0), args.integer(1), norm);
      else if (op == "rfft") r = at::fft_rfft(x, args.optional_integer(0), args.integer(1), norm);
      else if (op == "irfft") r = at::fft_irfft(x, args.optional_integer(0), args.integer(1), norm);
      else if (op == "hfft") r = at::fft_hfft(x, args.optional_integer(0), args.integer(1), norm);
      else if (op == "ihfft") r = at::fft_ihfft(x, args.optional_integer(0), args.integer(1), norm);
      else {
        const auto size_values = args.optional_integers(0);
        const auto dim_values = args.optional_integers(1);
        const auto sizes = optional_dims(size_values);
        const auto dims = optional_dims(dim_values);
        if (op == "fftn") r = at::fft_fftn(x, sizes, dims, norm);
        else if (op == "ifftn") r = at::fft_ifftn(x, sizes, dims, norm);
        else if (op == "rfftn") r = at::fft_rfftn(x, sizes, dims, norm);
        else if (op == "irfftn") r = at::fft_irfftn(x, sizes, dims, norm);
        else if (op == "hfftn") r = at::fft_hfftn(x, sizes, dims, norm);
        else if (op == "ihfftn") r = at::fft_ihfftn(x, sizes, dims, norm);
        else throw Error("ArgumentError", "unknown FFT kernel: " + op);
      }
    }
    *out = args.result(std::move(r));
  });
}
