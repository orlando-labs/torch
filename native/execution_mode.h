#pragma once
#include <c10/core/AutogradState.h>
#include <c10/core/GradMode.h>
#include <c10/core/InferenceMode.h>
#include <torch/version.h>
#include <cstdint>
#include <optional>

namespace sputnik_torch {
// The boundary validates mode; this guard receives only grad/no-grad/inference
// (0/1/2). Compare actual carrier-thread state, never a cached logical mode.
struct ExecutionModeGuard {
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 14
  c10::AutogradState previous = c10::AutogradState::get_tls_state();
  c10::impl::LocalDispatchKeySet previous_keys = c10::impl::tls_local_dispatch_key_set();
  bool changed_inference = false;
  bool changed_grad = false;

  explicit ExecutionModeGuard(int64_t mode) {
    const bool inferred = mode == 2;
    const auto excluded = previous_keys.excluded_ & c10::autograd_dispatch_keyset;
    // Match every field reset by this verified LibTorch constructor. A grad
    // toggle alone does not need to rewrite inference state or dispatch keys.
    changed_inference = previous.get_inference_mode() != inferred ||
        previous.get_fw_grad_mode() == inferred ||
        previous.get_multithreading_enabled() == inferred ||
        previous.get_view_replay_enabled() ||
        !previous.get_grad_layout_enforcement_enabled() ||
        previous.get_graph_exec_group().has_value() ||
        previous_keys.included_.has(c10::DispatchKey::ADInplaceOrView) == inferred ||
        excluded != (inferred ? c10::autograd_dispatch_keyset : c10::DispatchKeySet());
    changed_grad = previous.get_grad_mode() != (mode == 0);
    if (changed_inference) {
      // Same constructor effects as InferenceMode followed by AutoGradMode,
      // with the final grad bit installed in the single state write.
      c10::AutogradState::set_tls_state(c10::AutogradState(
          mode == 0, inferred, !inferred, !inferred));
      auto keys = previous_keys;
      keys.included_ = inferred
          ? keys.included_.remove(c10::DispatchKey::ADInplaceOrView)
          : keys.included_.add(c10::DispatchKey::ADInplaceOrView);
      keys.excluded_ = inferred
          ? keys.excluded_ | c10::autograd_dispatch_keyset
          : keys.excluded_ - c10::autograd_dispatch_keyset;
      c10::impl::_force_tls_local_dispatch_key_set(keys);
    } else if (changed_grad) {
      c10::AutogradState::get_tls_state().set_grad_mode(mode == 0);
    }
  }

  static bool same_state(const c10::AutogradState &a, const c10::AutogradState &b) {
    // Nonempty graph execution groups always take the restoring path: the
    // contained Python object's identity is deliberately not inspected here.
    return !a.get_graph_exec_group() && !b.get_graph_exec_group() &&
        a.get_grad_mode() == b.get_grad_mode() &&
        a.get_inference_mode() == b.get_inference_mode() &&
        a.get_fw_grad_mode() == b.get_fw_grad_mode() &&
        a.get_multithreading_enabled() == b.get_multithreading_enabled() &&
        a.get_view_replay_enabled() == b.get_view_replay_enabled() &&
        a.get_grad_layout_enforcement_enabled() == b.get_grad_layout_enforcement_enabled();
  }

  ~ExecutionModeGuard() {
    // The body may include callbacks or external C++ code. Even when entry was
    // a no-op, restore unexpected state changes just like the ordinary guards.
    if (!same_state(c10::AutogradState::get_tls_state(), previous))
      c10::AutogradState::set_tls_state(previous);
    const auto keys = c10::impl::tls_local_dispatch_key_set();
    if (keys.included_ != previous_keys.included_ || keys.excluded_ != previous_keys.excluded_)
      c10::impl::_force_tls_local_dispatch_key_set(previous_keys);
  }
#else
  // Other LibTorch versions retain the complete original guards until their
  // constructor effects have been verified.
  std::optional<c10::InferenceMode> inference;
  std::optional<c10::AutoGradMode> grad;
  explicit ExecutionModeGuard(int64_t mode) {
    inference.emplace(mode == 2);
    grad.emplace(mode == 0);
  }
#endif
  ExecutionModeGuard(const ExecutionModeGuard &) = delete;
  ExecutionModeGuard &operator=(const ExecutionModeGuard &) = delete;
};
} // namespace sputnik_torch
