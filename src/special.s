package torch.special

from torch.autograd.context import execution_mode
from torch.tensor.value import Tensor

export airy_ai, bessel_j0, bessel_j1, bessel_y0, bessel_y1, digamma
export entr, erf, erfc, erfcx, erfinv, expit
export expm1, exp2, gammaln, i0, i0e, i1
export i1e, log1p, log_ndtr, modified_bessel_i0, modified_bessel_i1, modified_bessel_k0
export modified_bessel_k1, ndtr, ndtri, psi, scaled_modified_bessel_k0, scaled_modified_bessel_k1
export sinc, spherical_bessel_j0, chebyshev_polynomial_t, chebyshev_polynomial_u, chebyshev_polynomial_v, chebyshev_polynomial_w
export hermite_polynomial_h, hermite_polynomial_he, laguerre_polynomial_l, legendre_polynomial_p, shifted_chebyshev_polynomial_t, shifted_chebyshev_polynomial_u
export shifted_chebyshev_polynomial_v, shifted_chebyshev_polynomial_w, xlog1py, xlogy, zeta, gammainc
export gammaincc, polygamma, multigammaln, logit, round, logsumexp
export softmax, log_softmax

native def _special(op as Str, inputs, options, mode as Int) from "amt_special"

def _kernel(op, inputs, options = []):
  handles = inputs.map: if Tensor === $it then $it._core else $it
  Tensor(_special(op, handles, options, execution_mode()))

def airy_ai(x): _kernel("airy_ai", [x])
def bessel_j0(x): _kernel("bessel_j0", [x])
def bessel_j1(x): _kernel("bessel_j1", [x])
def bessel_y0(x): _kernel("bessel_y0", [x])
def bessel_y1(x): _kernel("bessel_y1", [x])
def digamma(x): _kernel("digamma", [x])
def entr(x): _kernel("entr", [x])
def erf(x): _kernel("erf", [x])
def erfc(x): _kernel("erfc", [x])
def erfcx(x): _kernel("erfcx", [x])
def erfinv(x): _kernel("erfinv", [x])
def expit(x): _kernel("expit", [x])
def expm1(x): _kernel("expm1", [x])
def exp2(x): _kernel("exp2", [x])
def gammaln(x): _kernel("gammaln", [x])
def i0(x): _kernel("i0", [x])
def i0e(x): _kernel("i0e", [x])
def i1(x): _kernel("i1", [x])
def i1e(x): _kernel("i1e", [x])
def log1p(x): _kernel("log1p", [x])
def log_ndtr(x): _kernel("log_ndtr", [x])
def modified_bessel_i0(x): _kernel("modified_bessel_i0", [x])
def modified_bessel_i1(x): _kernel("modified_bessel_i1", [x])
def modified_bessel_k0(x): _kernel("modified_bessel_k0", [x])
def modified_bessel_k1(x): _kernel("modified_bessel_k1", [x])
def ndtr(x): _kernel("ndtr", [x])
def ndtri(x): _kernel("ndtri", [x])
def psi(x): _kernel("psi", [x])
def scaled_modified_bessel_k0(x): _kernel("scaled_modified_bessel_k0", [x])
def scaled_modified_bessel_k1(x): _kernel("scaled_modified_bessel_k1", [x])
def sinc(x): _kernel("sinc", [x])
def spherical_bessel_j0(x): _kernel("spherical_bessel_j0", [x])

def chebyshev_polynomial_t(x, n): _kernel("chebyshev_polynomial_t", [x, n])
def chebyshev_polynomial_u(x, n): _kernel("chebyshev_polynomial_u", [x, n])
def chebyshev_polynomial_v(x, n): _kernel("chebyshev_polynomial_v", [x, n])
def chebyshev_polynomial_w(x, n): _kernel("chebyshev_polynomial_w", [x, n])
def hermite_polynomial_h(x, n): _kernel("hermite_polynomial_h", [x, n])
def hermite_polynomial_he(x, n): _kernel("hermite_polynomial_he", [x, n])
def laguerre_polynomial_l(x, n): _kernel("laguerre_polynomial_l", [x, n])
def legendre_polynomial_p(x, n): _kernel("legendre_polynomial_p", [x, n])
def shifted_chebyshev_polynomial_t(x, n): _kernel("shifted_chebyshev_polynomial_t", [x, n])
def shifted_chebyshev_polynomial_u(x, n): _kernel("shifted_chebyshev_polynomial_u", [x, n])
def shifted_chebyshev_polynomial_v(x, n): _kernel("shifted_chebyshev_polynomial_v", [x, n])
def shifted_chebyshev_polynomial_w(x, n): _kernel("shifted_chebyshev_polynomial_w", [x, n])
def xlog1py(x, y): _kernel("xlog1py", [x, y])
def xlogy(x, y): _kernel("xlogy", [x, y])
def zeta(x, y): _kernel("zeta", [x, y])

def gammainc(x, y): _kernel("gammainc", [x, y])
def gammaincc(x, y): _kernel("gammaincc", [x, y])
def polygamma(n, x): _kernel("polygamma", [x], [n])
def multigammaln(x, p): _kernel("multigammaln", [x], [p])
def logit(x, eps: null): _kernel("logit", [x], [eps])
def round(x, decimals: 0): _kernel("round", [x], [decimals])

def logsumexp(x, dim, keepdim: false):
  dims = if Int === dim then [dim] else dim.array
  _kernel("logsumexp", [x], [dims, keepdim])

def softmax(x, dim:, dtype: null):
  _kernel("softmax", [x], [dim, (if dtype == null then null else dtype.str)])

def log_softmax(x, dim:, dtype: null):
  _kernel("log_softmax", [x], [dim, (if dtype == null then null else dtype.str)])

