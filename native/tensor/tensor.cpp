#include "bridge.h"
#include "kernel_args.h"
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <functional>
#include <torch/mps.h>
#include <type_traits>
#include <unordered_set>

using namespace sputnik_torch;
namespace {
at::Scalar scalar(SputnikCtx *cx, SputnikValue v) {
  int b;
  int64_t i;
  double f;
  if (sputnik_as_bool(cx, v, &b))
    return at::Scalar(bool(b));
  if (sputnik_as_int(cx, v, &i))
    return at::Scalar(i);
  if (sputnik_as_float(cx, v, &f))
    return at::Scalar(f);
  throw Error("TypeError", "expected a numeric scalar");
}
at::ScalarType dtype(const std::string &name) {
  return tensor_dtype(name);
}
std::string dtype_name(at::ScalarType type) {
  switch (type) {
  case at::kFloat:
    return "float32";
  case at::kDouble:
    return "float64";
  case at::kHalf:
    return "float16";
  case at::kBFloat16:
    return "bfloat16";
  case at::kComplexFloat:
    return "complex64";
  case at::kComplexDouble:
    return "complex128";
  case at::kLong:
    return "int64";
  case at::kInt:
    return "int32";
  case at::kShort:
    return "int16";
  case at::kChar:
    return "int8";
  case at::kByte:
    return "uint8";
  case at::kBool:
    return "bool";
  default:
    throw Error("Torch.Errors.DTypeError", "unsupported tensor dtype");
  }
}
at::Device device(const std::string &name) {
  return tensor_device(name);
}
void placement(at::ScalarType t, const at::Device &d) {
  require(!(d.is_mps() && (t == at::kDouble || t == at::kComplexDouble)),
          "Torch.Errors.DTypeError", "MPS does not support float64/complex128; cast explicitly");
}
void check_shape(const std::vector<int64_t> &dims) {
  int64_t count = 1;
  require(dims.size() <= 64, "Torch.Errors.ShapeError", "rank exceeds 64");
  for (auto d : dims) {
    require(d >= 0, "Torch.Errors.ShapeError", "negative factory dimension");
    require(d == 0 || count <= std::numeric_limits<int64_t>::max() / d,
            "Torch.Errors.ShapeError", "shape size overflows int64");
    count *= d;
  }
}
std::vector<int64_t> flatten(SputnikCtx *cx, SputnikValue v, std::vector<at::Scalar> &values,
                             size_t depth = 0) {
  require(depth <= 64, "Torch.Errors.ShapeError", "nested tensor input exceeds rank 64");
  if (!sputnik_is_list(cx, v)) {
    values.push_back(scalar(cx, v));
    return {};
  }
  const auto n = sputnik_list_len(cx, v);
  std::vector<int64_t> sub;
  for (size_t i = 0; i < n; ++i) {
    auto current = flatten(cx, sputnik_list_at(cx, v, i), values, depth + 1);
    if (i == 0)
      sub = current;
    else
      require(current == sub, "Torch.Errors.ShapeError", "ragged tensor input");
  }
  sub.insert(sub.begin(), static_cast<int64_t>(n));
  return sub;
}
template <class T> void fill_data(at::Tensor &t, const std::vector<at::Scalar> &values) {
  auto *data = t.data_ptr<T>();
  for (size_t i = 0; i < values.size(); ++i) {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      require(values[i].isIntegral(true), "Torch.Errors.DTypeError",
              "integer tensor construction requires integral values");
      auto value = values[i].toLong();
      require(value >= std::numeric_limits<T>::min() && value <= std::numeric_limits<T>::max(),
              "Torch.Errors.DTypeError", "scalar outside dtype range");
      data[i] = static_cast<T>(value);
    } else
      data[i] = values[i].to<T>();
  }
}
SputnikValue scalar_value(SputnikCtx *cx, const at::Tensor &t) {
  if (t.is_complex()) {
    const auto value = t.item<c10::complex<double>>();
    const SputnikValue parts[] = {sputnik_make_float(cx, value.real()),
                                  sputnik_make_float(cx, value.imag())};
    return sputnik_make_list(cx, parts, 2);
  }
  if (t.scalar_type() == at::kBool)
    return sputnik_make_bool(cx, t.item<bool>());
  if (at::isIntegralType(t.scalar_type(), false))
    return sputnik_make_int(cx, t.item<int64_t>());
  return sputnik_make_float(cx, t.item<double>());
}
SputnikValue array_value_fallback(SputnikCtx *cx, const at::Tensor &t) {
  if (t.dim() == 0)
    return scalar_value(cx, t);
  std::vector<SputnikValue> values;
  for (int64_t i = 0; i < t.size(0); ++i)
    values.push_back(array_value_fallback(cx, t.select(0, i)));
  return sputnik_make_list(cx, values.data(), values.size());
}

template <class T>
SputnikValue array_data(SputnikCtx *cx, const T *data, at::IntArrayRef sizes,
                      at::IntArrayRef strides, size_t axis = 0) {
  if (axis == sizes.size()) {
    if constexpr (std::is_same_v<T, bool>)
      return sputnik_make_bool(cx, *data);
    else if constexpr (std::is_integral_v<T>)
      return sputnik_make_int(cx, static_cast<int64_t>(*data));
    else
      return sputnik_make_float(cx, static_cast<double>(*data));
  }
  std::vector<SputnikValue> values;
  values.reserve(static_cast<size_t>(sizes[axis]));
  for (int64_t index = 0; index < sizes[axis]; ++index) {
    // Empty tensors can have a null data pointer and positive outer sizes.
    // Their inner empty dimension terminates recursion without dereferencing
    // or performing pointer arithmetic on that null pointer.
    const T *child = data == nullptr ? nullptr : data + index * strides[axis];
    values.push_back(array_data(cx, child, sizes, strides, axis + 1));
  }
  return sputnik_make_list(cx, values.data(), values.size());
}

SputnikValue array_value(SputnikCtx *cx, const at::Tensor &t) {
  // Materialization already owns a detached CPU tensor. Read its logical
  // elements directly, preserving storage offset and noncontiguous strides,
  // instead of dispatching select/item and creating a TensorImpl per element.
  // Less common layouts/dtypes and lazy conjugate/negative views keep their
  // existing LibTorch scalar conversion semantics.
  if (!t.is_cpu() || t.layout() != at::kStrided || t.is_conj() || t.is_neg())
    return array_value_fallback(cx, t);
  switch (t.scalar_type()) {
  case at::kBool: return array_data(cx, t.const_data_ptr<bool>(), t.sizes(), t.strides());
  case at::kByte: return array_data(cx, t.const_data_ptr<uint8_t>(), t.sizes(), t.strides());
  case at::kChar: return array_data(cx, t.const_data_ptr<int8_t>(), t.sizes(), t.strides());
  case at::kShort: return array_data(cx, t.const_data_ptr<int16_t>(), t.sizes(), t.strides());
  case at::kInt: return array_data(cx, t.const_data_ptr<int32_t>(), t.sizes(), t.strides());
  case at::kLong: return array_data(cx, t.const_data_ptr<int64_t>(), t.sizes(), t.strides());
  case at::kHalf: return array_data(cx, t.const_data_ptr<at::Half>(), t.sizes(), t.strides());
  case at::kBFloat16: return array_data(cx, t.const_data_ptr<at::BFloat16>(), t.sizes(), t.strides());
  case at::kFloat: return array_data(cx, t.const_data_ptr<float>(), t.sizes(), t.strides());
  case at::kDouble: return array_data(cx, t.const_data_ptr<double>(), t.sizes(), t.strides());
  default: return array_value_fallback(cx, t);
  }
}
} // namespace

extern "C" {
void amt_tensor_free(void *ptr) {
  delete static_cast<TensorCore *>(ptr);
}

// Metadata never submits kernels or materializes device data. These methods
// avoid execution-mode/task-local and string-operation dispatch on each read.
#define TENSOR_QUERY(name, expression) \
  SputnikStatus amt_tensor_##name(SputnikCtx *cx, SputnikValue self, const SputnikValue *, size_t n, SputnikValue *out) { \
    return boundary(cx, [&] { argc_check(n, 0); auto &c = core(cx, self); \
      const auto &t = c.tensor; *out = (expression); }); \
  }
TENSOR_QUERY(rank, sputnik_make_int(cx, t.dim()))
TENSOR_QUERY(numel, sputnik_make_int(cx, t.numel()))

// Optional typed leaf sidecar. The host has already checked the TensorCore tag
// and tombstone; no SputnikCtx/arena or result-builder callback is needed.
static SputnikStatus tensor_numel_leaf(const void *receiver, int64_t *out,
                                      const SputnikLeafFault *fault) noexcept {
  BoundaryProfileScope profile("amt_tensor_numel_leaf");
  try {
    *out = static_cast<const TensorCore *>(receiver)->tensor.numel();
    return SPUTNIK_OK;
  } catch (const Error &e) {
    return sputnik_leaf_fault(fault, error_class(e.kind), e.what());
  } catch (const c10::Error &e) {
    return sputnik_leaf_fault(fault, error_class("Torch.Errors.Error"), e.what_without_backtrace());
  } catch (const std::bad_alloc &) {
    return sputnik_leaf_fault(fault, error_class("Torch.Errors.Error"), "native allocation failed");
  } catch (const std::exception &e) {
    return sputnik_leaf_fault(fault, error_class("Torch.Errors.Error"), e.what());
  } catch (...) {
    return sputnik_leaf_fault(fault, error_class("Torch.Errors.Error"), "unknown native exception");
  }
}

extern "C" const SputnikLeafDescriptor *amt_tensor_numel_sputnik_leaf_v1() {
  static const SputnikLeafDescriptor descriptor{
      SPUTNIK_LEAF_ABI_VERSION, sizeof(SputnikLeafDescriptor),
      SPUTNIK_LEAF_HANDLE_INT, tensor_tag, {&tensor_numel_leaf}};
  return &descriptor;
}

static at::Tensor tensor_relu(const TensorCore &value, int64_t execution) {
  ExecutionGuard mode(execution);
  return value.tensor.relu();
}

SputnikStatus amt_tensor_relu(SputnikCtx *cx, SputnikValue self,
                             const SputnikValue *args, size_t argc,
                             SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(argc, 1);
    *out = wrap(cx, tensor_relu(core(cx, self), integer(cx, args[0])));
  });
}

static SputnikStatus tensor_relu_leaf(const void *receiver, int64_t execution,
                                      void **out, const SputnikLeafFault *fault) noexcept {
  BoundaryProfileScope profile("amt_tensor_relu_leaf");
  try {
    auto result = std::make_unique<TensorCore>(TensorCore{
        tensor_relu(*static_cast<const TensorCore *>(receiver), execution), false});
    *out = result.release();
    return SPUTNIK_OK;
  } catch (const Error &e) {
    return sputnik_leaf_fault(fault, error_class(e.kind), e.what());
  } catch (const c10::Error &e) {
    return sputnik_leaf_fault(fault, error_class("Torch.Errors.Error"), e.what_without_backtrace());
  } catch (const std::bad_alloc &) {
    return sputnik_leaf_fault(fault, error_class("Torch.Errors.Error"), "native allocation failed");
  } catch (const std::exception &e) {
    return sputnik_leaf_fault(fault, error_class("Torch.Errors.Error"), e.what());
  } catch (...) {
    return sputnik_leaf_fault(fault, error_class("Torch.Errors.Error"), "unknown native exception");
  }
}

extern "C" const SputnikLeafDescriptor *amt_tensor_relu_sputnik_leaf_v2() {
  static const SputnikLeafDescriptor descriptor = [] {
    SputnikLeafDescriptor result{};
    result.abi_version = SPUTNIK_LEAF_ABI_VERSION_2;
    result.struct_size = sizeof(SputnikLeafDescriptor);
    result.signature = SPUTNIK_LEAF_HANDLE_INT_HANDLE;
    result.receiver_tag = tensor_tag;
    result.handle_int_handle = &tensor_relu_leaf;
    return result;
  }();
  return &descriptor;
}

TENSOR_QUERY(dtype, text(cx, dtype_name(t.scalar_type())))
TENSOR_QUERY(device, text(cx, t.device().str()))
TENSOR_QUERY(requires_grad, sputnik_make_bool(cx, t.requires_grad()))
TENSOR_QUERY(leaf, sputnik_make_bool(cx, t.is_leaf()))
TENSOR_QUERY(readonly, sputnik_make_bool(cx, c.readonly))
TENSOR_QUERY(contiguous, sputnik_make_bool(cx, t.is_contiguous()))
TENSOR_QUERY(grad, t.grad().defined() ? wrap(cx, t.grad(), c.readonly) : sputnik_make_null(cx))
TENSOR_QUERY(identity, sputnik_make_int(cx, static_cast<int64_t>(reinterpret_cast<uintptr_t>(t.unsafeGetTensorImpl()))))
#undef TENSOR_QUERY
SputnikStatus amt_tensor_shape(SputnikCtx *cx, SputnikValue self, const SputnikValue *, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 0);
    std::vector<SputnikValue> values;
    for (auto size : core(cx, self).tensor.sizes()) values.push_back(sputnik_make_int(cx, size));
    *out = sputnik_make_list(cx, values.data(), values.size());
  });
}
SputnikStatus amt_tensor_strides(SputnikCtx *cx, SputnikValue self, const SputnikValue *, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 0);
    std::vector<SputnikValue> values;
    for (auto stride : core(cx, self).tensor.strides()) values.push_back(sputnik_make_int(cx, stride));
    *out = sputnik_make_list(cx, values.data(), values.size());
  });
}

SputnikStatus amt_info(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    auto key = string(cx, a[0]);
    if (key == "version")
      *out = text(cx, TORCH_VERSION);
    else if (key == "num_threads")
      *out = sputnik_make_int(cx, at::get_num_threads());
    else if (key == "num_interop_threads")
      *out = sputnik_make_int(cx, at::get_num_interop_threads());
    else if (key == "mps_built")
      *out = sputnik_make_bool(cx, at::hasMPS());
    else if (key == "mps_available")
      *out = sputnik_make_bool(cx, torch::mps::is_available());
    else if (key == "cuda_built")
      *out = sputnik_make_bool(cx, at::hasCUDA());
    else if (key == "cuda_available")
      *out = sputnik_make_bool(cx, at::getNumGPUs() > 0);
    else if (key == "cuda_device_count")
      *out = sputnik_make_int(cx, at::getNumGPUs());
    else if (key == "rocm_built")
      *out = sputnik_make_bool(cx, at::globalContext().hasROCM());
    else if (key == "synchronize") {
      synchronize_devices();
      *out = sputnik_make_null(cx);
    } else
      throw Error("ArgumentError", "unknown info key");
  });
}
SputnikStatus amt_synchronize(SputnikCtx *cx, const SputnikValue *, size_t n,
                              SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 0);
    synchronize_devices();
    *out = sputnik_make_null(cx);
  });
}
SputnikStatus amt_tensor(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 5);
    ExecutionGuard mode(integer(cx, a[4]));
    std::vector<at::Scalar> values;
    auto shape = flatten(cx, a[0], values);
    check_shape(shape);
    auto name = string(cx, a[1]);
    if (name == "auto") {
      bool ints = !values.empty(), bools = !values.empty();
      for (const auto &v : values) {
        ints = ints && v.isIntegral(true);
        bools = bools && v.isBoolean();
      }
      name = bools ? "bool" : ints ? "int64" : "float32";
    }
    auto type = dtype(name);
    auto dev = device(string(cx, a[2]));
    placement(type, dev);
    auto t = at::empty(shape, at::TensorOptions().dtype(type).device(at::kCPU));
    switch (type) {
    case at::kFloat:
      fill_data<float>(t, values);
      break;
    case at::kDouble:
      fill_data<double>(t, values);
      break;
    case at::kHalf:
      fill_data<at::Half>(t, values);
      break;
    case at::kBFloat16:
      fill_data<at::BFloat16>(t, values);
      break;
    case at::kComplexFloat:
      fill_data<c10::complex<float>>(t, values);
      break;
    case at::kComplexDouble:
      fill_data<c10::complex<double>>(t, values);
      break;
    case at::kLong:
      fill_data<int64_t>(t, values);
      break;
    case at::kInt:
      fill_data<int32_t>(t, values);
      break;
    case at::kShort:
      fill_data<int16_t>(t, values);
      break;
    case at::kChar:
      fill_data<int8_t>(t, values);
      break;
    case at::kByte:
      fill_data<uint8_t>(t, values);
      break;
    case at::kBool:
      fill_data<bool>(t, values);
      break;
    default:
      throw Error("Torch.Errors.DTypeError", "unsupported dtype");
    }
    t = t.to(dev);
    t.set_requires_grad(boolean(cx, a[3]));
    *out = wrap(cx, std::move(t));
  });
}
SputnikStatus amt_factory(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 7);
    ExecutionGuard mode(integer(cx, a[6]));
    auto op = string(cx, a[0]);
    auto shape = integers(cx, a[1]);
    check_shape(shape);
    auto type = dtype(string(cx, a[2]));
    auto dev = device(string(cx, a[3]));
    placement(type, dev);
    auto options = at::TensorOptions().dtype(type).device(at::kCPU);
    at::Tensor t;
    if (op == "zeros")
      t = at::zeros(shape, options);
    else if (op == "ones")
      t = at::ones(shape, options);
    else if (op == "empty")
      t = at::empty(shape, options);
    else if (op == "full")
      t = at::full(shape, scalar(cx, a[5]), options);
    else if (op == "fftfreq" || op == "rfftfreq") {
      require(shape.size() == 1 && shape[0] > 0, "ArgumentError", "frequency size must be positive");
      const auto spacing = number(cx, a[5]);
      require(std::isfinite(spacing) && spacing != 0, "ArgumentError", "frequency spacing must be finite and nonzero");
      t = op == "fftfreq" ? at::fft_fftfreq(shape[0], spacing, options)
                           : at::fft_rfftfreq(shape[0], spacing, options);
    }
    else if (op == "eye") {
      require(shape.size() == 2, "ArgumentError", "eye needs rows and columns");
      t = at::eye(shape[0], shape[1], options);
    } else if (op == "arange" || op == "linspace") {
      require(sputnik_is_list(cx, a[5]) && sputnik_list_len(cx, a[5]) == 3, "ArgumentError",
              "range factory requires three options");
      const auto start = scalar(cx, sputnik_list_at(cx, a[5], 0));
      const auto end = scalar(cx, sputnik_list_at(cx, a[5], 1));
      t = op == "arange"
              ? at::arange(start, end, scalar(cx, sputnik_list_at(cx, a[5], 2)), options)
              : at::linspace(start, end, integer(cx, sputnik_list_at(cx, a[5], 2)), options);
    } else if (op == "randn" || op == "rand") {
      auto g = at::detail::createCPUGenerator(static_cast<uint64_t>(integer(cx, a[5])));
      t = op == "randn" ? at::randn(shape, g, options) : at::rand(shape, g, options);
    } else
      throw Error("ArgumentError", "unknown factory");
    t = t.to(dev);
    t.set_requires_grad(boolean(cx, a[4]));
    *out = wrap(cx, std::move(t));
  });
}
SputnikStatus amt_meta(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 3);
    const auto execution = integer(cx, a[2]);
    ExecutionGuard mode(execution);
    auto &c = core(cx, a[0]);
    const auto &t = c.tensor;
    auto op = string(cx, a[1]);
    require(execution < 4 || (op != "item" && op != "to_a"), "Torch.Errors.ONNXError",
            "tensor data cannot be materialized as Sputnik scalars/arrays during ONNX export");
    if (op == "shape" || op == "strides") {
      std::vector<SputnikValue> values;
      auto data = op == "shape" ? t.sizes() : t.strides();
      for (auto v : data)
        values.push_back(sputnik_make_int(cx, v));
      *out = sputnik_make_list(cx, values.data(), values.size());
    } else if (op == "dtype")
      *out = text(cx, dtype_name(t.scalar_type()));
    else if (op == "device")
      *out = text(cx, t.device().str());
    else if (op == "numel")
      *out = sputnik_make_int(cx, t.numel());
    else if (op == "rank")
      *out = sputnik_make_int(cx, t.dim());
    else if (op == "requires_grad")
      *out = sputnik_make_bool(cx, t.requires_grad());
    else if (op == "leaf")
      *out = sputnik_make_bool(cx, t.is_leaf());
    else if (op == "readonly")
      *out = sputnik_make_bool(cx, c.readonly);
    else if (op == "contiguous")
      *out = sputnik_make_bool(cx, t.is_contiguous());
    else if (op == "sparse")
      *out = sputnik_make_bool(cx, t.is_sparse());
    else if (op == "grad")
      *out = t.grad().defined() ? wrap(cx, t.grad(), c.readonly) : sputnik_make_null(cx);
    else if (op == "identity")
      *out = sputnik_make_int(
          cx, static_cast<int64_t>(reinterpret_cast<uintptr_t>(t.unsafeGetTensorImpl())));
    else
      throw Error("ArgumentError", "unknown tensor metadata");
  });
}
SputnikStatus amt_materialize(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 3);
    const auto execution = integer(cx, a[2]);
    ExecutionGuard mode(execution);
    require(execution < 4, "Torch.Errors.ONNXError",
            "tensor data cannot be materialized as Sputnik scalars/arrays during ONNX export");
    const auto &t = core(cx, a[0]).tensor;
    auto op = string(cx, a[1]);
    if (op == "item") {
      require(t.numel() == 1, "Torch.Errors.ShapeError", "item requires exactly one element");
      *out = scalar_value(cx, t);
    } else if (op == "to_a")
      *out = array_value(cx, t.detach().cpu());
    else
      throw Error("ArgumentError", "unknown tensor materialization");
  });
}
SputnikStatus amt_binary(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]));
    auto &x = core(cx, a[0]);
    auto op = string(cx, a[2]);
    at::Tensor result;
    if (!sputnik_is_handle(cx, a[1])) {
      auto y = scalar(cx, a[1]);
      if (op == "add")
        result = x.tensor + y;
      else if (op == "sub")
        result = x.tensor - y;
      else if (op == "mul")
        result = x.tensor * y;
      else if (op == "div")
        result = x.tensor / y;
      else if (op == "eq")
        result = x.tensor.eq(y);
      else if (op == "ne")
        result = x.tensor.ne(y);
      else if (op == "lt")
        result = x.tensor.lt(y);
      else if (op == "le")
        result = x.tensor.le(y);
      else if (op == "gt")
        result = x.tensor.gt(y);
      else if (op == "ge")
        result = x.tensor.ge(y);
      else if (op == "pow")
        result = x.tensor.pow(y);
      else
        throw Error("TypeError", "operation requires another Tensor");
      *out = wrap(cx, std::move(result));
      return;
    }
    auto y = core(cx, a[1]).tensor;
    if (op == "add")
      result = x.tensor + y;
    else if (op == "sub")
      result = x.tensor - y;
    else if (op == "mul")
      result = x.tensor * y;
    else if (op == "div")
      result = x.tensor / y;
    else if (op == "matmul")
      result = x.tensor.matmul(y);
    else if (op == "complex")
      result = at::complex(x.tensor, y);
    else if (op == "polar")
      result = at::polar(x.tensor, y);
    else if (op == "eq")
      result = x.tensor.eq(y);
    else if (op == "ne")
      result = x.tensor.ne(y);
    else if (op == "lt")
      result = x.tensor.lt(y);
    else if (op == "le")
      result = x.tensor.le(y);
    else if (op == "gt")
      result = x.tensor.gt(y);
    else if (op == "ge")
      result = x.tensor.ge(y);
    else if (op == "pow")
      result = x.tensor.pow(y);
    else
      throw Error("ArgumentError", "unknown binary operation");
    *out = wrap(cx, std::move(result));
  });
}
SputnikStatus amt_unary(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]));
    auto &c = core(cx, a[0]);
    auto &x = c.tensor;
    auto op = string(cx, a[1]);
    at::Tensor r;
    bool ro = false;
    if (op == "sum")
      r = x.sum();
    else if (op == "mean")
      r = x.mean();
    else if (op == "square")
      r = x.square();
    else if (op == "relu")
      r = x.relu();
    else if (op == "sigmoid")
      r = x.sigmoid();
    else if (op == "tanh")
      r = x.tanh();
    else if (op == "neg")
      r = x.neg();
    else if (op == "abs")
      r = x.abs();
    else if (op == "exp")
      r = x.exp();
    else if (op == "log")
      r = x.log();
    else if (op == "sqrt")
      r = x.sqrt();
    else if (op == "rsqrt")
      r = x.rsqrt();
    else if (op == "sin")
      r = x.sin();
    else if (op == "cos")
      r = x.cos();
    else if (op == "real") { r = at::real(x); ro = c.readonly; }
    else if (op == "imag") { r = at::imag(x); ro = c.readonly; }
    else if (op == "conj") { r = x.conj(); ro = c.readonly; }
    else if (op == "resolve_conj") { r = x.resolve_conj(); ro = c.readonly; }
    else if (op == "angle") r = at::angle(x);
    else if (op == "view_as_real") { r = at::view_as_real(x); ro = c.readonly; }
    else if (op == "view_as_complex") { r = at::view_as_complex(x); ro = c.readonly; }
    else if (op == "clamp") {
      require(sputnik_is_list(cx, a[2]) && sputnik_list_len(cx, a[2]) == 2, "ArgumentError",
              "clamp needs bounds");
      r = x.clamp(scalar(cx, sputnik_list_at(cx, a[2], 0)), scalar(cx, sputnik_list_at(cx, a[2], 1)));
    } else if (op == "sum_dim" || op == "mean_dim") {
      auto dim = integer(cx, sputnik_list_at(cx, a[2], 0));
      auto keep = boolean(cx, sputnik_list_at(cx, a[2], 1));
      r = op == "sum_dim" ? x.sum({dim}, keep) : x.mean({dim}, keep);
    } else if (op == "argmax") {
      auto dim = sputnik_list_at(cx, a[2], 0);
      r = x.argmax(sputnik_is_null(cx, dim) ? std::nullopt : std::optional<int64_t>(integer(cx, dim)),
                   boolean(cx, sputnik_list_at(cx, a[2], 1)));
    } else if (op == "index_select") {
      auto dim = integer(cx, sputnik_list_at(cx, a[2], 0));
      auto &indices = core(cx, sputnik_list_at(cx, a[2], 1)).tensor;
      r = x.index_select(dim, indices);
    } else if (op == "softmax")
      r = x.softmax(integer(cx, a[2]));
    else if (op == "log_softmax")
      r = x.log_softmax(integer(cx, a[2]));
    else if (op == "flatten") {
      auto dims = integers(cx, a[2]);
      require(dims.size() == 2, "ArgumentError", "flatten needs two axes");
      r = x.flatten(dims[0], dims[1]);
      ro = c.readonly;
    } else if (op == "permute") {
      r = x.permute(integers(cx, a[2]));
      ro = c.readonly;
    } else if (op == "unsqueeze") {
      r = x.unsqueeze(integer(cx, a[2]));
      ro = c.readonly;
    } else if (op == "squeeze") {
      r = sputnik_is_null(cx, a[2]) ? x.squeeze() : x.squeeze(integer(cx, a[2]));
      ro = c.readonly;
    } else if (op == "contiguous") {
      r = x.contiguous();
      ro = c.readonly;
    } else if (op == "to_dense") {
      r = x.to_dense();
      ro = c.readonly && x.layout() == at::kStrided;
    } else if (op == "copy")
      r = x.clone();
    else if (op == "snapshot") {
      r = x.detach().clone();
      ro = true;
    } else if (op == "detach") {
      r = x.detach();
      ro = c.readonly;
    } else if (op == "reshape") {
      r = x.reshape(integers(cx, a[2]));
      ro = c.readonly;
    } else if (op == "transpose") {
      auto d = integers(cx, a[2]);
      require(d.size() == 2, "ArgumentError", "transpose needs two axes");
      r = x.transpose(d[0], d[1]);
      ro = c.readonly;
    } else if (op == "select") {
      auto d = integers(cx, a[2]);
      require(d.size() == 2, "ArgumentError", "select needs dim and index");
      r = x.select(d[0], d[1]);
      ro = c.readonly;
    } else if (op == "narrow") {
      auto dims = integers(cx, a[2]);
      require(dims.size() == 3, "ArgumentError", "narrow needs axis, start and length");
      r = x.narrow(dims[0], dims[1], dims[2]);
      ro = c.readonly;
    } else
      throw Error("ArgumentError", "unknown unary operation");
    *out = wrap(cx, std::move(r), ro);
  });
}
SputnikStatus amt_to(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]));
    auto &c = core(cx, a[0]);
    auto dev = device(string(cx, a[1]));
    auto name = string(cx, a[2]);
    auto type = name == "same" ? c.tensor.scalar_type() : dtype(name);
    placement(type, dev);
    *out = wrap(cx, c.tensor.to(dev, type), c.readonly);
  });
}
SputnikStatus amt_mutate(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
    ExecutionGuard mode(integer(cx, a[3]), true);
    auto &c = core(cx, a[0]);
    writable(c);
    auto &t = c.tensor;
    auto op = string(cx, a[1]);
    if (op == "zero")
      t.zero_();
    else if (op == "fill")
      t.fill_(scalar(cx, a[2]));
    else if (op == "requires_grad")
      t.set_requires_grad(boolean(cx, a[2]));
    else if (op == "copy_from") {
      auto &source = core(cx, a[2]).tensor;
      require(t.sizes() == source.sizes(), "Torch.Errors.ShapeError",
              "copy_from! requires the same shape");
      require(t.scalar_type() == source.scalar_type(), "Torch.Errors.DTypeError",
              "copy_from! requires the same dtype");
      require(t.device() == source.device(), "Torch.Errors.DeviceError",
              "copy_from! requires the same device; use to explicitly");
      t.copy_(source);
    } else if (op == "set_grad") {
      if (sputnik_is_null(cx, a[2])) {
        t.mutable_grad() = at::Tensor();
      } else {
        const auto &gradient = core(cx, a[2]).tensor;
        require(t.sizes() == gradient.sizes() && t.scalar_type() == gradient.scalar_type() &&
                    t.device() == gradient.device() && gradient.layout() == at::kStrided,
                "Torch.Errors.AutogradError", "grad! requires matching dense values");
        at::NoGradGuard guard;
        t.mutable_grad() = gradient.detach().clone();
      }
    } else if (op == "zero_grad") {
      if (boolean(cx, a[2]))
        t.mutable_grad() = at::Tensor();
      else if (t.grad().defined()) {
        at::NoGradGuard g;
        t.mutable_grad().zero_();
      }
    } else if (op == "add" || op == "sub" || op == "mul") {
      if (sputnik_is_handle(cx, a[2])) {
        auto &y = core(cx, a[2]).tensor;
        if (op == "add")
          t.add_(y);
        else if (op == "sub")
          t.sub_(y);
        else
          t.mul_(y);
      } else {
        auto y = scalar(cx, a[2]);
        if (op == "add")
          t.add_(y);
        else if (op == "sub")
          t.sub_(y);
        else
          t.mul_(y);
      }
    } else
      throw Error("ArgumentError", "unknown mutation");
    *out = a[0];
  });
}
SputnikStatus amt_backward(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 5);
    require(integer(cx, a[4]) < 4, "Torch.Errors.ONNXError", "backward is forbidden during ONNX export");
    at::AutoGradMode mode(true);
    auto &c = core(cx, a[0]);
    writable(c);
    auto gradient = sputnik_is_null(cx, a[1]) ? at::Tensor() : core(cx, a[1]).tensor;
    c.tensor.backward(gradient, boolean(cx, a[2]), boolean(cx, a[3]));
    *out = a[0];
  });
}
SputnikStatus amt_load_parameter(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 3);
    ExecutionGuard mode(integer(cx, a[2]), true);
    auto &dst = core(cx, a[0]);
    auto &src = core(cx, a[1]);
    writable(dst);
    require(dst.tensor.sizes() == src.tensor.sizes() &&
                dst.tensor.scalar_type() == src.tensor.scalar_type(),
            "Torch.Errors.ShapeError", "parameter shape/dtype mismatch");
    at::NoGradGuard guard;
    dst.tensor.copy_(src.tensor);
    *out = a[0];
  });
}
}
