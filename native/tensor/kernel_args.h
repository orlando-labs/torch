#pragma once
#include "bridge.h"

namespace sputnik_torch {
inline at::ScalarType tensor_dtype(const std::string &name) {
  if (name == "float32") return at::kFloat;
  if (name == "float64") return at::kDouble;
  if (name == "float16") return at::kHalf;
  if (name == "bfloat16") return at::kBFloat16;
  if (name == "complex64") return at::kComplexFloat;
  if (name == "complex128") return at::kComplexDouble;
  if (name == "int64") return at::kLong;
  if (name == "int32") return at::kInt;
  if (name == "int16") return at::kShort;
  if (name == "int8") return at::kChar;
  if (name == "uint8") return at::kByte;
  if (name == "bool") return at::kBool;
  throw Error("Torch.Errors.DTypeError", "unsupported dtype: " + name);
}
inline at::OptionalIntArrayRef optional_dims(const std::optional<std::vector<int64_t>> &dims) {
  return dims ? at::OptionalIntArrayRef(*dims) : at::OptionalIntArrayRef(std::nullopt);
}
// Shared decoding for curated kernels. Array-backed option references own
// their storage for the entire LibTorch call.
struct KernelArgs {
  SputnikCtx *cx;
  SputnikValue inputs, options;
  KernelArgs(SputnikCtx *cx, SputnikValue inputs, SputnikValue options)
      : cx(cx), inputs(inputs), options(options) {
    require(sputnik_is_list(cx, inputs) && sputnik_is_list(cx, options),
            "TypeError", "kernel inputs and options must be arrays");
  }
  SputnikValue input(size_t i) const {
    require(i < sputnik_list_len(cx, inputs), "ArgumentError", "missing kernel input");
    return sputnik_list_at(cx, inputs, i);
  }
  SputnikValue arg(size_t i) const {
    require(i < sputnik_list_len(cx, options), "ArgumentError", "missing kernel option");
    return sputnik_list_at(cx, options, i);
  }
  at::Tensor tensor(size_t i) const { return core(cx, input(i)).tensor; }
  int64_t integer(size_t i) const { return sputnik_torch::integer(cx, arg(i)); }
  double number(size_t i) const { return sputnik_torch::number(cx, arg(i)); }
  bool boolean(size_t i) const { return sputnik_torch::boolean(cx, arg(i)); }
  std::string string(size_t i) const { return sputnik_torch::string(cx, arg(i)); }
  std::vector<int64_t> integers(size_t i) const { return sputnik_torch::integers(cx, arg(i)); }
  std::optional<int64_t> optional_integer(size_t i) const {
    return sputnik_is_null(cx, arg(i)) ? std::nullopt : std::optional<int64_t>(integer(i));
  }
  std::optional<double> optional_number(size_t i) const {
    return sputnik_is_null(cx, arg(i)) ? std::nullopt : std::optional<double>(number(i));
  }
  std::optional<at::ScalarType> optional_dtype(size_t i) const {
    return sputnik_is_null(cx, arg(i)) ? std::nullopt
                                     : std::optional<at::ScalarType>(tensor_dtype(string(i)));
  }
  std::optional<std::vector<int64_t>> optional_integers(size_t i) const {
    return sputnik_is_null(cx, arg(i)) ? std::nullopt
                                     : std::optional<std::vector<int64_t>>(integers(i));
  }
  SputnikValue result(at::Tensor value) const {
    bool readonly = false;
    for (size_t i = 0; i < sputnik_list_len(cx, inputs); ++i) {
      auto v = input(i);
      if (!sputnik_is_handle(cx, v)) continue;
      const auto &c = core(cx, v);
      readonly |= c.readonly && value.is_alias_of(c.tensor);
    }
    return wrap(cx, std::move(value), readonly);
  }
  template <class... T> SputnikValue result(const std::tuple<T...> &values) const {
    std::vector<SputnikValue> wrapped;
    std::apply([&](const auto &...value) { (wrapped.push_back(result(value)), ...); }, values);
    return sputnik_make_list(cx, wrapped.data(), wrapped.size());
  }
};
inline at::Scalar numeric_scalar(SputnikCtx *cx, SputnikValue value) {
  int64_t i;
  int b;
  if (sputnik_as_bool(cx, value, &b)) return at::Scalar(bool(b));
  if (sputnik_as_int(cx, value, &i)) return at::Scalar(i);
  return at::Scalar(number(cx, value));
}
} // namespace sputnik_torch
