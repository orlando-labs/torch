#include "bridge.h"
#include <ATen/CPUGeneratorImpl.h>
#include <mutex>

using namespace sputnik_torch;
namespace {
constexpr char tag[] = "torch.GeneratorCore";
struct GeneratorCore {
  at::Generator generator;
};
} // namespace
extern "C" void amt_generator_free(void *ptr) {
  delete static_cast<GeneratorCore *>(ptr);
}
extern "C" SputnikStatus amt_generator_new(SputnikCtx *cx, const SputnikValue *a, size_t n,
                                         SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    auto p = std::make_unique<GeneratorCore>(
        GeneratorCore{at::detail::createCPUGenerator(static_cast<uint64_t>(integer(cx, a[0])))});
    *out = sputnik_make_handle(cx, tag, p.get());
    if (!*out)
      throw FaultPending{};
    p.release();
  });
}
extern "C" SputnikStatus amt_generator(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 5);
    void *raw = nullptr;
    if (!sputnik_handle_ptr(cx, a[0], tag, &raw))
      throw FaultPending{};
    auto &g = static_cast<GeneratorCore *>(raw)->generator;
    const auto op = string(cx, a[1]);
    at::NoGradGuard guard;
    if (op == "state") {
      *out = wrap(cx, g.get_state().clone(), true);
      return;
    }
    if (op == "restore") {
      g.set_state(core(cx, a[2]).tensor);
      *out = a[0];
      return;
    }
    if (op == "seed") {
      std::lock_guard<std::mutex> lock(g.mutex());
      g.set_current_seed(static_cast<uint64_t>(integer(cx, a[2])));
      *out = a[0];
      return;
    }
    const auto dims = integers(cx, a[2]);
    require(dims.size() <= 64, "Torch.Errors.ShapeError", "random rank exceeds 64");
    for (auto d : dims)
      require(d >= 0, "Torch.Errors.ShapeError", "negative random dimension");
    const auto dtype = string(cx, a[3]);
    require(dtype == "float32" || dtype == "float64", "Torch.Errors.DTypeError",
            "Generator samples require float32 or float64");
    const auto device = tensor_device(string(cx, a[4]));
    require(!(device.is_mps() && dtype == "float64"), "Torch.Errors.DTypeError",
            "MPS cannot store float64");
    auto options =
        at::TensorOptions().dtype(dtype == "float64" ? at::kDouble : at::kFloat).device(at::kCPU);
    at::Tensor result;
    if (op == "rand")
      result = at::rand(dims, g, options);
    else if (op == "randn")
      result = at::randn(dims, g, options);
    else if (op == "randperm") {
      require(dims.size() == 1, "ArgumentError", "randperm needs one size");
      result = at::randperm(dims[0], g, options.dtype(at::kLong));
    } else
      throw Error("ArgumentError", "unknown generator operation");
    *out = wrap(cx, result.to(device));
  });
}
