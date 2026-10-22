#include "group.h"
#include <cstdlib>
#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#endif
using namespace sputnik_torch;
extern "C" {
void amt_group_free(SputnikCtx *, void *ptr) {
  delete static_cast<GroupCore *>(ptr);
}
SputnikStatus amt_gloo_available(SputnikCtx *cx, const SputnikValue *, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 0);
#ifdef USE_C10D_GLOO
    *out = sputnik_make_bool(cx, 1);
#else
    *out=sputnik_make_bool(cx,0);
#endif
  });
}
SputnikStatus amt_group_open(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 5);
#ifdef USE_C10D_GLOO
    auto rank = integer(cx, a[0]), size = integer(cx, a[1]), ms = integer(cx, a[3]);
    require(size > 0 && size <= INT_MAX && rank >= 0 && rank < size, "ArgumentError",
            "invalid rank/world_size");
    require(ms > 0 && ms <= 3600000, "ArgumentError", "timeout_ms must be in 1..3600000");
    auto p = std::make_unique<GroupCore>();
    p->timeout = std::chrono::milliseconds(ms);
    const auto path = string(cx, a[2]);
    require(!path.empty(), "ArgumentError", "rendezvous file path must not be empty");
    p->store = c10::make_intrusive<c10d::FileStore>(path, static_cast<int>(size));
    p->store->setTimeout(p->timeout);
    auto options = c10d::ProcessGroupGloo::Options::create(p->timeout);
    options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForHostname(string(cx, a[4])));
    p->group = c10::make_intrusive<c10d::ProcessGroupGloo>(p->store, static_cast<int>(rank),
                                                           static_cast<int>(size), options);
    *out = sputnik_make_handle(cx, group_tag, p.get());
    if (!*out)
      throw FaultPending{};
    p.release();
#else
    throw Error("Torch.Errors.DistributedError","adapter was built without matching Gloo headers");
#endif
  });
}
SputnikStatus amt_collective(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 4);
#ifdef USE_C10D_GLOO
    void *raw = nullptr;
    if (!sputnik_handle_ptr(cx, a[0], group_tag, &raw))
      throw FaultPending{};
    auto &g = *static_cast<GroupCore *>(raw);
    require(!g.failed, "Torch.Errors.DistributedError",
            "process group has failed; recreate worker group");
    auto op = string(cx, a[1]);
    if (op == "rank") {
      *out = sputnik_make_int(cx, g.group->getRank());
      return;
    }
    if (op == "world_size") {
      *out = sputnik_make_int(cx, g.group->getSize());
      return;
    }
    if (sputnik_task_cancelled(cx))
      throw Error("CancelledError", "collective cancelled before launch");
    std::vector<at::Tensor> values;
    if (op != "barrier") {
      auto &t = core(cx, a[2]);
      writable(t);
      require(t.tensor.device().is_cpu() && t.tensor.layout() == at::kStrided &&
                  t.tensor.is_contiguous(),
              "Torch.Errors.DistributedError", "Gloo requires contiguous dense CPU tensors");
      require(!t.tensor.requires_grad(), "Torch.Errors.DistributedError",
              "collectives require detached tensors");
      values.push_back(t.tensor);
    }
    try {
      c10::intrusive_ptr<c10d::Work> work;
      if (op == "barrier")
        work = g.group->barrier();
      else if (op == "all_reduce")
        work = g.group->allreduce(values);
      else if (op == "broadcast") {
        auto rank = integer(cx, a[3]);
        require(rank >= 0 && rank < g.group->getSize(), "ArgumentError",
                "broadcast source outside group");
        c10d::BroadcastOptions opts;
        opts.rootRank = rank;
        work = g.group->broadcast(values, opts);
      } else
        throw Error("ArgumentError", "unknown collective");
      require(work->wait(g.timeout), "Torch.Errors.DistributedError",
              "collective did not complete");
    } catch (...) {
      g.failed = true;
      throw;
    }
    *out = a[0];
#else
    throw Error("Torch.Errors.DistributedError","Gloo is unavailable");
#endif
  });
}
SputnikStatus amt_group_env(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    auto env = [](const char *name) {
      auto value = std::getenv(name);
      require(value != nullptr, "Torch.Errors.DistributedError",
              std::string("missing launcher variable: ") + name);
      return std::string(value);
    };
    auto env_integer = [&](const char *name) {
      auto value = env(name);
      size_t used = 0;
      auto number = std::stoll(value, &used);
      require(used == value.size(), "ArgumentError", "invalid rank environment value");
      return number;
    };
    SputnikValue args[] = {sputnik_make_int(cx, env_integer("RANK")),
                         sputnik_make_int(cx, env_integer("WORLD_SIZE")),
                         text(cx, env("SPUTNIK_RENDEZVOUS")), a[0], text(cx, "127.0.0.1")};
    if (amt_group_open(cx, args, 5, out) != SPUTNIK_OK)
      throw FaultPending{};
  });
}
}
