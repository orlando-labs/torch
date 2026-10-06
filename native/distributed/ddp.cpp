#include "group.h"
#include <sstream>
#include <unordered_set>
using namespace sputniktorch;
#ifdef USE_C10D_GLOO
namespace {
void wait(GroupCore &g, c10::intrusive_ptr<c10d::Work> work) {
  try {
    require(work->wait(g.timeout), "Sputniktorch.Errors.DistributedError",
            "DDP collective timed out");
  } catch (...) {
    g.failed = true;
    throw;
  }
}
void broadcast(GroupCore &g, at::Tensor t) {
  std::vector<at::Tensor> values{std::move(t)};
  wait(g, g.group->broadcast(values));
}
void sum(GroupCore &g, at::Tensor t) {
  std::vector<at::Tensor> values{std::move(t)};
  wait(g, g.group->allreduce(values));
}
void validate_schema(GroupCore &g, const std::string &schema) {
  const bool root = g.group->getRank() == 0;
  auto length = at::full({1}, root ? int64_t(schema.size()) : 0, at::kLong);
  broadcast(g, length);
  const auto n = length.item<int64_t>();
  require(n >= 0 && n <= 1024 * 1024, "Sputniktorch.Errors.DistributedError",
          "DDP schema exceeds 1 MiB");
  auto bytes = at::zeros({n}, at::kByte);
  if (root && n)
    std::memcpy(bytes.data_ptr(), schema.data(), n);
  broadcast(g, bytes);
  std::string reference(static_cast<char *>(bytes.data_ptr()), n);
  auto mismatch = at::full({1}, schema == reference ? 0 : 1, at::kLong);
  sum(g, mismatch);
  require(mismatch.item<int64_t>() == 0, "Sputniktorch.Errors.DistributedError",
          "DDP parameter schemas differ across ranks");
}
} // namespace
#endif
extern "C" SputnikStatus amt_ddp_parameters(SputnikCtx *cx, const SputnikValue *a, size_t n,
                                          SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 3);
#ifdef USE_C10D_GLOO
    void *raw = nullptr;
    if (!sputnik_handle_ptr(cx, a[0], group_tag, &raw))
      throw FaultPending{};
    auto &g = *static_cast<GroupCore *>(raw);
    require(!g.failed, "Sputniktorch.Errors.DistributedError", "process group has failed");
    require(sputnik_is_list(cx, a[1]), "TypeError", "expected named parameter entries");
    const auto op = string(cx, a[2]);
    require(op == "sync" || op == "reduce", "ArgumentError", "invalid DDP operation");
    std::vector<at::Tensor> parameters;
    std::unordered_set<const void *> seen;
    std::ostringstream schema;
    for (size_t i = 0; i < sputnik_list_len(cx, a[1]); ++i) {
      auto entry = sputnik_list_at(cx, a[1], i);
      require(sputnik_is_list(cx, entry) && sputnik_list_len(cx, entry) == 2, "TypeError",
              "invalid parameter entry");
      auto name = string(cx, sputnik_list_at(cx, entry, 0));
      auto &c = core(cx, sputnik_list_at(cx, entry, 1));
      writable(c);
      auto &p = c.tensor;
      require(p.device().is_cpu() && p.is_leaf() && p.is_contiguous() &&
                  p.layout() == at::kStrided && p.is_floating_point(),
              "Sputniktorch.Errors.DistributedError",
              "DDP requires contiguous dense floating CPU leaf parameters");
      if (!seen.insert(p.unsafeGetTensorImpl()).second)
        continue;
      schema << name.size() << ':' << name << ':' << int(p.scalar_type()) << ':' << p.sizes() << ':'
             << p.requires_grad() << ';';
      parameters.push_back(p);
    }
    require(schema.str().size() <= 1024 * 1024, "Sputniktorch.Errors.DistributedError",
            "DDP schema exceeds 1 MiB");
    validate_schema(g, schema.str());
    at::NoGradGuard guard;
    if (op == "sync") {
      for (auto &p : parameters)
        broadcast(g, p);
    } else {
      // One common usage vector keeps collective order identical with locally
      // unused parameters; gradients are averaged over the full world size.
      auto usage = at::zeros({int64_t(parameters.size())}, at::kLong);
      for (size_t i = 0; i < parameters.size(); ++i) {
        if (parameters[i].grad().defined())
          usage.data_ptr<int64_t>()[i] = 1;
      }
      if (!parameters.empty())
        sum(g, usage);
      for (size_t i = 0; i < parameters.size(); ++i) {
        if (!usage.data_ptr<int64_t>()[i])
          continue;
        auto &p = parameters[i];
        auto grad = p.grad().defined() ? p.grad().detach().contiguous() : at::zeros_like(p);
        sum(g, grad);
        grad.div_(g.group->getSize());
        p.mutable_grad() = std::move(grad);
      }
    }
    *out = sputnik_make_null(cx);
#else
    throw Error("Sputniktorch.Errors.DistributedError", "Gloo is unavailable");
#endif
  });
}
