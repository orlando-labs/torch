#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <time.h>

namespace sputnik_torch {
// Opt-in diagnostic only. CPU time separates host work from MPS waits; nested
// callbacks have exclusive totals so execution_batch does not double count
// every kernel. Timings include argument validation and the LibTorch call.
struct BoundaryProfileTotals {
  uint64_t calls = 0, wall_ns = 0, cpu_ns = 0;
};
struct BoundaryProfiles {
  std::mutex mutex;
  std::map<std::string, BoundaryProfileTotals> totals;
  ~BoundaryProfiles() {
    for (const auto &[name, value] : totals) {
      std::fprintf(stderr,
          "AMT_BOUNDARY_PROFILE {\"function\":\"%s\",\"calls\":%llu,"
          "\"exclusive_wall_ns\":%llu,\"exclusive_cpu_ns\":%llu}\n",
          name.c_str(), static_cast<unsigned long long>(value.calls),
          static_cast<unsigned long long>(value.wall_ns),
          static_cast<unsigned long long>(value.cpu_ns));
    }
  }
};
inline bool boundary_profile_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("AMT_BOUNDARY_PROFILE");
    return value != nullptr && std::string(value) == "1";
  }();
  return enabled;
}
inline BoundaryProfiles &boundary_profiles() {
  static BoundaryProfiles profiles;
  return profiles;
}
struct BoundaryProfileScope {
  inline static thread_local BoundaryProfileScope *current = nullptr;
  const char *name;
  BoundaryProfileScope *parent = nullptr;
  uint64_t wall_start = 0, cpu_start = 0, child_wall = 0, child_cpu = 0;
  bool enabled;
  static uint64_t wall_time() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  static uint64_t cpu_time() {
    timespec value{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value);
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL + value.tv_nsec;
  }
  explicit BoundaryProfileScope(const char *name)
      : name(name), enabled(boundary_profile_enabled()) {
    if (!enabled) return;
    parent = current;
    current = this;
    wall_start = wall_time();
    cpu_start = cpu_time();
  }
  ~BoundaryProfileScope() {
    if (!enabled) return;
    const uint64_t cpu = cpu_time() - cpu_start;
    const uint64_t wall = wall_time() - wall_start;
    current = parent;
    auto &profiles = boundary_profiles();
    {
      std::lock_guard<std::mutex> lock(profiles.mutex);
      auto &total = profiles.totals[name];
      ++total.calls;
      total.wall_ns += wall - child_wall;
      total.cpu_ns += cpu - child_cpu;
    }
    // Include bookkeeping as child work in the parent's subtraction. It is
    // profiling overhead, not the Sputnik callback's exclusive runtime cost.
    if (parent != nullptr) {
      parent->child_wall += wall_time() - wall_start;
      parent->child_cpu += cpu_time() - cpu_start;
    }
  }
};
} // namespace sputnik_torch
