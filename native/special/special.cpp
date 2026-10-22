#include "tensor/kernel_args.h"

using namespace sputnik_torch;
extern "C" SputnikStatus amt_special(SputnikCtx *cx, const SputnikValue *a, size_t n,
                                      SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]));
    KernelArgs k(cx, a[1], a[2]);
    const auto op = string(cx, a[0]);
    at::Tensor r;
    if (op == "airy_ai") r = at::special_airy_ai(k.tensor(0));
    else if (op == "bessel_j0") r = at::special_bessel_j0(k.tensor(0));
    else if (op == "bessel_j1") r = at::special_bessel_j1(k.tensor(0));
    else if (op == "bessel_y0") r = at::special_bessel_y0(k.tensor(0));
    else if (op == "bessel_y1") r = at::special_bessel_y1(k.tensor(0));
    else if (op == "digamma") r = at::special_digamma(k.tensor(0));
    else if (op == "entr") r = at::special_entr(k.tensor(0));
    else if (op == "erf") r = at::special_erf(k.tensor(0));
    else if (op == "erfc") r = at::special_erfc(k.tensor(0));
    else if (op == "erfcx") r = at::special_erfcx(k.tensor(0));
    else if (op == "erfinv") r = at::special_erfinv(k.tensor(0));
    else if (op == "expit") r = at::special_expit(k.tensor(0));
    else if (op == "expm1") r = at::special_expm1(k.tensor(0));
    else if (op == "exp2") r = at::special_exp2(k.tensor(0));
    else if (op == "gammaln") r = at::special_gammaln(k.tensor(0));
    else if (op == "i0") r = at::special_i0(k.tensor(0));
    else if (op == "i0e") r = at::special_i0e(k.tensor(0));
    else if (op == "i1") r = at::special_i1(k.tensor(0));
    else if (op == "i1e") r = at::special_i1e(k.tensor(0));
    else if (op == "log1p") r = at::special_log1p(k.tensor(0));
    else if (op == "log_ndtr") r = at::special_log_ndtr(k.tensor(0));
    else if (op == "modified_bessel_i0") r = at::special_modified_bessel_i0(k.tensor(0));
    else if (op == "modified_bessel_i1") r = at::special_modified_bessel_i1(k.tensor(0));
    else if (op == "modified_bessel_k0") r = at::special_modified_bessel_k0(k.tensor(0));
    else if (op == "modified_bessel_k1") r = at::special_modified_bessel_k1(k.tensor(0));
    else if (op == "ndtr") r = at::special_ndtr(k.tensor(0));
    else if (op == "ndtri") r = at::special_ndtri(k.tensor(0));
    else if (op == "psi") r = at::special_psi(k.tensor(0));
    else if (op == "scaled_modified_bessel_k0") r = at::special_scaled_modified_bessel_k0(k.tensor(0));
    else if (op == "scaled_modified_bessel_k1") r = at::special_scaled_modified_bessel_k1(k.tensor(0));
    else if (op == "sinc") r = at::special_sinc(k.tensor(0));
    else if (op == "spherical_bessel_j0") r = at::special_spherical_bessel_j0(k.tensor(0));
    else if (op == "chebyshev_polynomial_t") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_chebyshev_polynomial_t(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_chebyshev_polynomial_t(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_chebyshev_polynomial_t(k.tensor(0), k.tensor(1));
    }
    else if (op == "chebyshev_polynomial_u") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_chebyshev_polynomial_u(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_chebyshev_polynomial_u(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_chebyshev_polynomial_u(k.tensor(0), k.tensor(1));
    }
    else if (op == "chebyshev_polynomial_v") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_chebyshev_polynomial_v(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_chebyshev_polynomial_v(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_chebyshev_polynomial_v(k.tensor(0), k.tensor(1));
    }
    else if (op == "chebyshev_polynomial_w") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_chebyshev_polynomial_w(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_chebyshev_polynomial_w(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_chebyshev_polynomial_w(k.tensor(0), k.tensor(1));
    }
    else if (op == "hermite_polynomial_h") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_hermite_polynomial_h(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_hermite_polynomial_h(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_hermite_polynomial_h(k.tensor(0), k.tensor(1));
    }
    else if (op == "hermite_polynomial_he") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_hermite_polynomial_he(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_hermite_polynomial_he(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_hermite_polynomial_he(k.tensor(0), k.tensor(1));
    }
    else if (op == "laguerre_polynomial_l") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_laguerre_polynomial_l(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_laguerre_polynomial_l(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_laguerre_polynomial_l(k.tensor(0), k.tensor(1));
    }
    else if (op == "legendre_polynomial_p") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_legendre_polynomial_p(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_legendre_polynomial_p(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_legendre_polynomial_p(k.tensor(0), k.tensor(1));
    }
    else if (op == "shifted_chebyshev_polynomial_t") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_shifted_chebyshev_polynomial_t(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_shifted_chebyshev_polynomial_t(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_shifted_chebyshev_polynomial_t(k.tensor(0), k.tensor(1));
    }
    else if (op == "shifted_chebyshev_polynomial_u") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_shifted_chebyshev_polynomial_u(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_shifted_chebyshev_polynomial_u(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_shifted_chebyshev_polynomial_u(k.tensor(0), k.tensor(1));
    }
    else if (op == "shifted_chebyshev_polynomial_v") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_shifted_chebyshev_polynomial_v(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_shifted_chebyshev_polynomial_v(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_shifted_chebyshev_polynomial_v(k.tensor(0), k.tensor(1));
    }
    else if (op == "shifted_chebyshev_polynomial_w") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_shifted_chebyshev_polynomial_w(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_shifted_chebyshev_polynomial_w(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_shifted_chebyshev_polynomial_w(k.tensor(0), k.tensor(1));
    }
    else if (op == "xlog1py") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_xlog1py(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_xlog1py(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_xlog1py(k.tensor(0), k.tensor(1));
    }
    else if (op == "xlogy") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_xlogy(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_xlogy(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_xlogy(k.tensor(0), k.tensor(1));
    }
    else if (op == "zeta") {
      if (!sputnik_is_handle(cx, k.input(0)))
        r = at::special_zeta(numeric_scalar(cx, k.input(0)), k.tensor(1));
      else if (!sputnik_is_handle(cx, k.input(1)))
        r = at::special_zeta(k.tensor(0), numeric_scalar(cx, k.input(1)));
      else r = at::special_zeta(k.tensor(0), k.tensor(1));
    }
    else if (op == "gammainc") r = at::special_gammainc(k.tensor(0), k.tensor(1));
    else if (op == "gammaincc") r = at::special_gammaincc(k.tensor(0), k.tensor(1));
    else if (op == "polygamma") r = at::special_polygamma(k.integer(0), k.tensor(0));
    else if (op == "multigammaln") r = at::special_multigammaln(k.tensor(0), k.integer(0));
    else if (op == "logit") r = at::special_logit(k.tensor(0), k.optional_number(0));
    else if (op == "round") r = at::special_round(k.tensor(0), k.integer(0));
    else if (op == "logsumexp") r = at::special_logsumexp(k.tensor(0), k.integers(0), k.boolean(1));
    else if (op == "softmax") r = at::special_softmax(k.tensor(0), k.integer(0), k.optional_dtype(1));
    else if (op == "log_softmax") r = at::special_log_softmax(k.tensor(0), k.integer(0), k.optional_dtype(1));
    else throw Error("ArgumentError", "unknown special kernel: " + op);
    *out = k.result(std::move(r));
  });
}

