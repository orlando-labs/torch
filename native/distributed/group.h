#pragma once
#include "bridge.h"
#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#endif
namespace sputniktorch {
constexpr char group_tag[] = "sputniktorch.GroupCore";
struct GroupCore {
#ifdef USE_C10D_GLOO
  c10::intrusive_ptr<c10d::Store> store;
  c10::intrusive_ptr<c10d::ProcessGroupGloo> group;
#endif
  std::chrono::milliseconds timeout{20000};
  bool failed = false;
};
} // namespace sputniktorch
