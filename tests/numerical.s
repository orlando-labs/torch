package torch.numerical_test

import torch.fft as fft
import torch.linalg as la
import torch.special as special
from torch import tensor, zeros, ones, eye, complex, polar, no_grad
from torch.checks import check

export main

def close(x, y, tolerance: 1e-8):
  check(x.shape == y.shape and (x - y).abs.sum.item < tolerance, "numerical result")

def main():
  x = tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype: "float64", requires_grad: true)
  spectrum = fft.fft(x)
  check(spectrum.dtype.complex? and spectrum.dtype.byte_size == 16, "complex dtype")
  close(fft.ifft(spectrum).real, x)
  check(spectrum.to_a[0] == [15.0, 0.0], "complex materialization")
  check(spectrum[0].item == [15.0, 0.0], "complex scalar materialization")
  [:backward, :forward, :ortho].each |norm|:
    close(fft.ifft(fft.fft(x, norm:), norm:).real, x)
    close(fft.irfft(fft.rfft(x, norm:), n: 5, norm:), x)
    close(fft.hfft(fft.ihfft(x, norm:), n: 5, norm:), x)
  check(fft.fft(x, n: 7).shape == [7], "FFT padding")
  close(fft.fftshift(fft.ifftshift(x)), x)
  close(fft.fftfreq(4, d: 0.5, dtype: "float64"), tensor([0.0, 0.5, -1.0, -0.5], dtype: "float64"))
  close(fft.rfftfreq(4, dtype: "float64"), tensor([0.0, 0.25, 0.5], dtype: "float64"))
  matrix = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype: "float64")
  close(fft.ifft2(fft.fft2(matrix)).real, matrix)
  close(fft.irfft2(fft.rfft2(matrix), s: [2, 3]), matrix)
  close(fft.hfft2(fft.ihfft2(matrix), s: [2, 3]), matrix)
  close(fft.ifftn(fft.fftn(matrix.transpose(0, 1)), dim: [0, 1]).real, matrix.transpose(0, 1))
  close(fft.irfftn(fft.rfftn(matrix), s: [2, 3]), matrix)
  close(fft.hfftn(fft.ihfftn(matrix), s: [2, 3]), matrix)
  spectrum.abs.square.sum.backward!
  close(x.grad, x.detach * 10)

  z = complex(matrix, matrix * 2)
  close(z.real, matrix)
  close(z.imag, matrix * 2)
  close(z.conj.imag, matrix * -2)
  close(z.view_as_real.view_as_complex, z)
  close(polar(ones([2], dtype: "float64"), zeros([2], dtype: "float64")).real, ones([2], dtype: "float64"))
  frozen = z.snapshot
  refused = false
  try:
    frozen.real.zero!
  rescue Torch.Errors.ReadOnlyTensorError:
    refused = true
  check(refused, "complex view preserves read-only")
  fresh = frozen.copy
  fresh.imag.zero!
  close(frozen.imag, matrix * 2)
  close(fresh.imag, zeros([2, 3], dtype: "float64"))

  a = tensor([[4.0, 1.0], [1.0, 3.0]], dtype: "float64", requires_grad: true)
  b = tensor([[1.0], [2.0]], dtype: "float64")
  identity = eye(2, dtype: "float64")
  close(a.matmul(la.solve(a, b)), b)
  close(a.matmul(la.inv(a)), identity)
  close(a.matmul(la.pinv(a)), identity)
  l = la.cholesky(a)
  close(l.matmul(l.transpose(-1, -2)), a)
  factors, info = la.cholesky_ex(a)
  check(info.item == 0, "cholesky status")
  close(factors, l)
  inverse, inverse_info = la.inv_ex(a)
  check(inverse_info.item == 0, "inverse status")
  close(inverse, la.inv(a))
  solution, solve_info = la.solve_ex(a, b)
  check(solve_info.item == 0, "solve status")
  close(solution, la.solve(a, b))
  close(l.matmul(la.solve_triangular(l, b, upper: false)), b)
  check((la.det(a).item - 11.0).abs < 1e-8, "determinant")
  sign, logdet = la.slogdet(a)
  check(sign.item == 1.0 and (logdet.exp.item - 11.0).abs < 1e-8, "slogdet")
  values, vectors = la.eigh(a)
  close(a.matmul(vectors), vectors * values)
  close(la.eigvalsh(a), values)
  eigenvalues, eigenvectors = la.eig(a)
  close(a.to(device: "cpu", dtype: "complex128").matmul(eigenvectors), eigenvectors * eigenvalues)
  close(la.eigvals(a), eigenvalues)
  u, s, vh = la.svd(a, full_matrices: false)
  close((u * s).matmul(vh), a)
  close(la.svdvals(a), s)
  q, r = la.qr(a)
  close(q.matmul(r), a)
  permutation, lower, upper = la.lu(a)
  close(permutation.matmul(lower).matmul(upper), a)
  lu, pivots = la.lu_factor(a)
  close(a.matmul(la.lu_solve(lu, pivots, b)), b)
  lu_ex, pivots_ex, lu_info = la.lu_factor_ex(a)
  check(lu_info.item == 0, "LU status")
  ld, ld_pivots = la.ldl_factor(a)
  close(a.matmul(la.ldl_solve(ld, ld_pivots, b)), b)
  ld_ex, ld_pivots_ex, ld_info = la.ldl_factor_ex(a)
  check(ld_info.item == 0, "LDL status")
  least_solution, residuals, rank, singular = la.lstsq(a, b, driver: "gelsd")
  close(a.matmul(least_solution), b)
  check(rank.item == 2 and singular.shape == [2], "least squares metadata")
  check(la.matrix_rank(a).item == 2, "matrix rank")
  check((la.vector_norm(tensor([3.0, 4.0], dtype: "float64")).item - 5.0).abs < 1e-8, "vector norm")
  check((la.matrix_norm(a).item - 27.0 ** 0.5).abs < 1e-8, "matrix norm")
  close(la.norm(a), la.matrix_norm(a))
  check(la.cond(identity).item == 1.0, "condition number")
  close(la.matrix_power(a, 2), a.matmul(a))
  close(la.matrix_exp(zeros([2, 2], dtype: "float64")), identity)
  root = la.matrix_sqrth(a)
  close(root.matmul(root), a)
  close(la.multi_dot([a, identity, a]), a.matmul(a))
  close(la.matmul(a, b), a.matmul(b))
  close(la.diagonal(a), tensor([4.0, 3.0], dtype: "float64"))
  close(la.cross(tensor([1.0, 0.0, 0.0]), tensor([0.0, 1.0, 0.0])), tensor([0.0, 0.0, 1.0]))
  check(la.vecdot(z, z, dim: 1).imag.abs.sum.item == 0.0, "Hermitian dot")
  check(la.vander(tensor([1.0, 2.0]), n: 3).to_a == [[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]], "Vandermonde")
  close(la.tensorinv(identity, ind: 1), identity)
  close(la.tensorsolve(identity, tensor([1.0, 2.0], dtype: "float64")), tensor([1.0, 2.0], dtype: "float64"))
  la.solve(a, b).sum.backward!
  check(a.grad.shape == a.shape, "solve autograd")

  v = tensor([0.0, 0.5, 1.0], dtype: "float64", requires_grad: true)
  close(special.erf(v) + special.erfc(v), ones([3], dtype: "float64"))
  close(special.ndtri(special.ndtr(v)), v)
  close(special.log_ndtr(v).exp, special.ndtr(v))
  close(special.expit(v), v.sigmoid)
  close(special.logit(special.expit(v)), v)
  close(special.expm1(v), v.exp - 1)
  close(special.exp2(v), (v * Math.log(2.0)).exp)
  close(special.log1p(v), (v + 1).log)
  close(special.psi(v + 1), special.digamma(v + 1))
  check(special.sinc(v)[0].item == 1.0, "sinc zero limit")
  close(special.gammainc(v + 1, v + 1) + special.gammaincc(v + 1, v + 1), ones([3], dtype: "float64"))
  close(special.xlogy(0, v), zeros([3], dtype: "float64"))
  close(special.xlog1py(v, 1), v * Math.log(2.0))
  close(special.chebyshev_polynomial_t(v, 2), v.square * 2 - 1)
  close(special.hermite_polynomial_h(v, 2), v.square * 4 - 2)
  close(special.legendre_polynomial_p(v, 2), (v.square * 3 - 1) / 2)
  special.erf(v).sum.backward!
  close(v.grad, v.detach.square.neg.exp * (2.0 / Math.sqrt(Math.PI)))
  print "PASS torch numerical CPU"
  0
