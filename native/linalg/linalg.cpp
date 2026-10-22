#include "tensor/kernel_args.h"

using namespace sputnik_torch;
extern "C" SputnikStatus amt_linalg(SputnikCtx *cx, const SputnikValue *a, size_t n,
                                     SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]));
    KernelArgs k(cx, a[1], a[2]);
    const auto op = string(cx, a[0]);
    const auto x = k.tensor(0);
    at::Tensor r;
    if (op == "cholesky") r = at::linalg_cholesky(x, k.boolean(0));
    else if (op == "inv") r = at::linalg_inv(x);
    else if (op == "det") r = at::linalg_det(x);
    else if (op == "eigvals") r = at::linalg_eigvals(x);
    else if (op == "eigvalsh") r = at::linalg_eigvalsh(x, k.string(0));
    else if (op == "matrix_exp") r = at::linalg_matrix_exp(x);
    else if (op == "matrix_sqrth") r = at::linalg_matrix_sqrth(x);
    else if (op == "matrix_power") r = at::linalg_matrix_power(x, k.integer(0));
    else if (op == "solve") r = at::linalg_solve(x, k.tensor(1), k.boolean(0));
    else if (op == "solve_triangular")
      r = at::linalg_solve_triangular(x, k.tensor(1), k.boolean(0), k.boolean(1), k.boolean(2));
    else if (op == "lu_solve")
      r = at::linalg_lu_solve(x, k.tensor(1), k.tensor(2), k.boolean(0), k.boolean(1));
    else if (op == "ldl_solve") r = at::linalg_ldl_solve(x, k.tensor(1), k.tensor(2), k.boolean(0));
    else if (op == "householder_product") r = at::linalg_householder_product(x, k.tensor(1));
    else if (op == "cross") r = at::linalg_cross(x, k.tensor(1), k.integer(0));
    else if (op == "vecdot") r = at::linalg_vecdot(x, k.tensor(1), k.integer(0));
    else if (op == "matmul") r = at::linalg_matmul(x, k.tensor(1));
    else if (op == "diagonal") r = at::linalg_diagonal(x, k.integer(0), k.integer(1), k.integer(2));
    else if (op == "vander") r = at::linalg_vander(x, k.optional_integer(0));
    else if (op == "tensorinv") r = at::linalg_tensorinv(x, k.integer(0));
    else if (op == "tensorsolve") {
      const auto dim_values = k.optional_integers(0);
      const auto dims = optional_dims(dim_values);
      r = at::linalg_tensorsolve(x, k.tensor(1), dims);
    } else if (op == "multi_dot") {
      std::vector<at::Tensor> inputs;
      for (size_t i = 0; i < sputnik_list_len(cx, a[1]); ++i) inputs.push_back(k.tensor(i));
      r = at::linalg_multi_dot(inputs);
    } else if (op == "pinv" || op == "matrix_rank") {
      if (sputnik_is_handle(cx, k.arg(0)) || sputnik_is_handle(cx, k.arg(1))) {
        auto tolerance = [&](size_t i) -> std::optional<at::Tensor> {
          if (sputnik_is_null(cx, k.arg(i))) return std::nullopt;
          if (sputnik_is_handle(cx, k.arg(i))) return core(cx, k.arg(i)).tensor;
          return at::scalar_tensor(number(cx, k.arg(i)), x.options().dtype(at::toRealValueType(x.scalar_type())));
        };
        r = op == "pinv" ? at::linalg_pinv(x, tolerance(0), tolerance(1), k.boolean(2))
                          : at::linalg_matrix_rank(x, tolerance(0), tolerance(1), k.boolean(2));
      } else {
        r = op == "pinv" ? at::linalg_pinv(x, k.optional_number(0), k.optional_number(1), k.boolean(2))
                          : at::linalg_matrix_rank(x, k.optional_number(0), k.optional_number(1), k.boolean(2));
      }
    } else if (op == "cond") {
      if (sputnik_is_str(cx, k.arg(0))) r = at::linalg_cond(x, k.string(0));
      else r = at::linalg_cond(x, sputnik_is_null(cx, k.arg(0)) ? std::nullopt
                               : std::optional<at::Scalar>(numeric_scalar(cx, k.arg(0))));
    } else if (op == "norm" || op == "vector_norm" || op == "matrix_norm") {
      const auto dim_values = k.optional_integers(1);
      const auto dims = optional_dims(dim_values);
      const auto dtype = k.optional_dtype(3);
      if (op == "vector_norm")
        r = at::linalg_vector_norm(x, numeric_scalar(cx, k.arg(0)), dims, k.boolean(2), dtype);
      else if (op == "matrix_norm") {
        require(dims.has_value(), "ArgumentError", "matrix_norm needs two dimensions");
        r = sputnik_is_str(cx, k.arg(0))
                ? at::linalg_matrix_norm(x, k.string(0), *dims, k.boolean(2), dtype)
                : at::linalg_matrix_norm(x, numeric_scalar(cx, k.arg(0)), *dims, k.boolean(2), dtype);
      } else if (sputnik_is_str(cx, k.arg(0)))
        r = at::linalg_norm(x, k.string(0), dims, k.boolean(2), dtype);
      else r = at::linalg_norm(x, sputnik_is_null(cx, k.arg(0)) ? std::nullopt
                                  : std::optional<at::Scalar>(numeric_scalar(cx, k.arg(0))),
                               dims, k.boolean(2), dtype);
    } else if (op == "svd" || op == "svdvals" || op == "lstsq") {
      const auto driver_index = op == "svdvals" ? 0U : 1U;
      const auto driver = sputnik_is_null(cx, k.arg(driver_index))
                              ? std::optional<std::string>() : std::optional<std::string>(k.string(driver_index));
      const auto driver_view = driver ? std::optional<c10::string_view>(*driver) : std::nullopt;
      if (op == "svd") { *out = k.result(at::linalg_svd(x, k.boolean(0), driver_view)); return; }
      if (op == "lstsq") { *out = k.result(at::linalg_lstsq(x, k.tensor(1), k.optional_number(0), driver_view)); return; }
      r = at::linalg_svdvals(x, driver_view);
    } else if (op == "eig") { *out = k.result(at::linalg_eig(x)); return; }
    else if (op == "eigh") { *out = k.result(at::linalg_eigh(x, k.string(0))); return; }
    else if (op == "qr") { *out = k.result(at::linalg_qr(x, k.string(0))); return; }
    else if (op == "slogdet") { *out = k.result(at::linalg_slogdet(x)); return; }
    else if (op == "cholesky_ex") { *out = k.result(at::linalg_cholesky_ex(x, k.boolean(0), k.boolean(1))); return; }
    else if (op == "inv_ex") { *out = k.result(at::linalg_inv_ex(x, k.boolean(0))); return; }
    else if (op == "solve_ex") { *out = k.result(at::linalg_solve_ex(x, k.tensor(1), k.boolean(0), k.boolean(1))); return; }
    else if (op == "lu") { *out = k.result(at::linalg_lu(x, k.boolean(0))); return; }
    else if (op == "lu_factor") { *out = k.result(at::linalg_lu_factor(x, k.boolean(0))); return; }
    else if (op == "lu_factor_ex") { *out = k.result(at::linalg_lu_factor_ex(x, k.boolean(0), k.boolean(1))); return; }
    else if (op == "ldl_factor") { *out = k.result(at::linalg_ldl_factor(x, k.boolean(0))); return; }
    else if (op == "ldl_factor_ex") { *out = k.result(at::linalg_ldl_factor_ex(x, k.boolean(0), k.boolean(1))); return; }
    else throw Error("ArgumentError", "unknown linalg kernel: " + op);
    *out = k.result(std::move(r));
  });
}
