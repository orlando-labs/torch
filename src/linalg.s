package torch.linalg

from torch.autograd.context import execution_mode
from torch.tensor.value import Tensor

export cholesky, cholesky_ex, inv, inv_ex, det, slogdet, eig, eigvals, eigh, eigvalsh
export svd, svdvals, qr, lu, lu_factor, lu_factor_ex, lu_solve, ldl_factor, ldl_factor_ex, ldl_solve
export solve, solve_ex, solve_triangular, lstsq, pinv, matrix_rank, norm, vector_norm, matrix_norm, cond
export matrix_exp, matrix_power, matrix_sqrth, multi_dot, matmul, cross, vecdot, diagonal
export householder_product, tensorinv, tensorsolve, vander

native def _linalg(op as Str, inputs, options, mode as Int) from "amt_linalg"

def _operand(x): if Tensor === x then x._core else x
def _dims(dim): if dim == null then null else (if Int === dim then [dim] else dim.array)
def _dtype(dtype): if dtype == null then null else dtype.str

def _kernel(op, inputs, options = []):
  handles = inputs.map: $it._core
  Tensor(_linalg(op, handles, options, execution_mode()))

def _tuple(op, inputs, options = []):
  inputs = inputs.map: $it._core
  handles = _linalg(op, inputs, options, execution_mode())
  values = handles.map: Tensor($it)
  values.tuple

def cholesky(x, upper: false): _kernel("cholesky", [x], [upper])
def cholesky_ex(x, upper: false, check_errors: false): _tuple("cholesky_ex", [x], [upper, check_errors])
def inv(x): _kernel("inv", [x])
def inv_ex(x, check_errors: false): _tuple("inv_ex", [x], [check_errors])
def det(x): _kernel("det", [x])
def slogdet(x): _tuple("slogdet", [x])
def eig(x): _tuple("eig", [x])
def eigvals(x): _kernel("eigvals", [x])
def eigh(x, uplo: "L"): _tuple("eigh", [x], [uplo.str])
def eigvalsh(x, uplo: "L"): _kernel("eigvalsh", [x], [uplo.str])
def svd(x, full_matrices: true, driver: null): _tuple("svd", [x], [full_matrices, driver])
def svdvals(x, driver: null): _kernel("svdvals", [x], [driver])
def qr(x, mode: :reduced): _tuple("qr", [x], [mode.str])
def lu(x, pivot: true): _tuple("lu", [x], [pivot])
def lu_factor(x, pivot: true): _tuple("lu_factor", [x], [pivot])
def lu_factor_ex(x, pivot: true, check_errors: false): _tuple("lu_factor_ex", [x], [pivot, check_errors])
def lu_solve(lu, pivots, b, left: true, adjoint: false): _kernel("lu_solve", [lu, pivots, b], [left, adjoint])
def ldl_factor(x, hermitian: false): _tuple("ldl_factor", [x], [hermitian])
def ldl_factor_ex(x, hermitian: false, check_errors: false): _tuple("ldl_factor_ex", [x], [hermitian, check_errors])
def ldl_solve(ld, pivots, b, hermitian: false): _kernel("ldl_solve", [ld, pivots, b], [hermitian])
def solve(a, b, left: true): _kernel("solve", [a, b], [left])
def solve_ex(a, b, left: true, check_errors: false): _tuple("solve_ex", [a, b], [left, check_errors])
def solve_triangular(a, b, upper:, left: true, unitriangular: false):
  _kernel("solve_triangular", [a, b], [upper, left, unitriangular])

def lstsq(a, b, rcond: null, driver: null): _tuple("lstsq", [a, b], [rcond, driver])
def pinv(x, atol: null, rtol: null, hermitian: false):
  _kernel("pinv", [x], [_operand(atol), _operand(rtol), hermitian])

def matrix_rank(x, atol: null, rtol: null, hermitian: false):
  _kernel("matrix_rank", [x], [_operand(atol), _operand(rtol), hermitian])

def norm(x, ord: null, dim: null, keepdim: false, dtype: null):
  order = if Symbol === ord then ord.str else ord
  _kernel("norm", [x], [order, _dims(dim), keepdim, _dtype(dtype)])

def vector_norm(x, ord: 2, dim: null, keepdim: false, dtype: null):
  _kernel("vector_norm", [x], [ord, _dims(dim), keepdim, _dtype(dtype)])

def matrix_norm(x, ord: :fro, dim: [-2, -1], keepdim: false, dtype: null):
  order = if Symbol === ord then ord.str else ord
  _kernel("matrix_norm", [x], [order, _dims(dim), keepdim, _dtype(dtype)])

def cond(x, p: null): _kernel("cond", [x], [(if Symbol === p then p.str else p)])
def matrix_exp(x): _kernel("matrix_exp", [x])
def matrix_sqrth(x): _kernel("matrix_sqrth", [x])
def matrix_power(x, n): _kernel("matrix_power", [x], [n])
def multi_dot(tensors): _kernel("multi_dot", tensors)
def matmul(x, y): _kernel("matmul", [x, y])
def cross(x, y, dim: -1): _kernel("cross", [x, y], [dim])
def vecdot(x, y, dim: -1): _kernel("vecdot", [x, y], [dim])
def diagonal(x, offset: 0, dim1: -2, dim2: -1): _kernel("diagonal", [x], [offset, dim1, dim2])
def householder_product(x, tau): _kernel("householder_product", [x, tau])
def tensorinv(x, ind: 2): _kernel("tensorinv", [x], [ind])
def tensorsolve(a, b, dims: null): _kernel("tensorsolve", [a, b], [_dims(dims)])
def vander(x, n: null): _kernel("vander", [x], [n])
