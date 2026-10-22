#include "execution_mode.h"
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
struct Snapshot {
  c10::AutogradState state = c10::AutogradState::get_tls_state();
  c10::impl::LocalDispatchKeySet keys = c10::impl::tls_local_dispatch_key_set();
  void restore() const {
    c10::AutogradState::set_tls_state(state);
    c10::impl::_force_tls_local_dispatch_key_set(keys);
  }
  bool matches() const {
    const Snapshot now;
    return state.get_grad_mode() == now.state.get_grad_mode() &&
        state.get_inference_mode() == now.state.get_inference_mode() &&
        state.get_fw_grad_mode() == now.state.get_fw_grad_mode() &&
        state.get_multithreading_enabled() == now.state.get_multithreading_enabled() &&
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 14
        state.get_view_replay_enabled() == now.state.get_view_replay_enabled() &&
        state.get_grad_layout_enforcement_enabled() == now.state.get_grad_layout_enforcement_enabled() &&
        state.get_graph_exec_group().has_value() == now.state.get_graph_exec_group().has_value() &&
#endif
        keys.included_ == now.keys.included_ && keys.excluded_ == now.keys.excluded_;
  }
};
void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void verify() {
  const Snapshot original;
  // Include states produced by external guards, not only our own cached mode.
  for (unsigned bits = 0; bits < 64; ++bits) {
    original.restore();
    c10::AutogradState state((bits & 1) != 0, (bits & 2) != 0,
                            (bits & 4) != 0, (bits & 8) != 0);
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 14
    state.set_view_replay_enabled((bits & 16) != 0);
    state.set_grad_layout_enforcement_enabled((bits & 32) == 0);
#endif
    c10::AutogradState::set_tls_state(state);
    auto keys = original.keys;
    if (bits & 2) {
      keys.included_ = keys.included_.remove(c10::DispatchKey::ADInplaceOrView);
      keys.excluded_ = keys.excluded_ | c10::autograd_dispatch_keyset;
    }
    // Deliberately include partially excluded Autograd dispatch state.
    if (bits & 16) keys.excluded_ = keys.excluded_.add(c10::DispatchKey::AutogradCPU);
    c10::impl::_force_tls_local_dispatch_key_set(keys);
    const Snapshot before;
    for (int mode = 0; mode < 3; ++mode) {
      before.restore();
      Snapshot expected;
      {
        c10::InferenceMode inference(mode == 2);
        c10::AutoGradMode grad(mode == 0);
        expected = Snapshot();
      }
      check(before.matches(), "reference guards restore external state");
      {
        sputnik_torch::ExecutionModeGuard guard(mode);
        check(expected.matches(), "optimized state differs from ordinary guards");
        for (int nested = 0; nested < 3; ++nested) {
          { sputnik_torch::ExecutionModeGuard inner(nested); }
          check(expected.matches(), "nested guard fails to restore outer state");
        }
      }
      check(before.matches(), "optimized guards leak carrier-thread state");
      try {
        sputnik_torch::ExecutionModeGuard guard(mode);
        throw 1;
      } catch (int) {}
      check(before.matches(), "exception unwind fails to restore state");
      {
        sputnik_torch::ExecutionModeGuard guard(mode);
        c10::GradMode::set_enabled(mode != 0);
        c10::AutogradState::get_tls_state().set_fw_grad_mode(false);
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 14
        c10::AutogradState::get_tls_state().set_view_replay_enabled(true);
#endif
        auto modified_keys = c10::impl::tls_local_dispatch_key_set();
        modified_keys.excluded_ = modified_keys.excluded_.add(c10::DispatchKey::AutogradCPU);
        c10::impl::_force_tls_local_dispatch_key_set(modified_keys);
      }
      check(before.matches(), "guard fails to restore state changed by its body");
    }
  }
  original.restore();
#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 14
  {
    sputnik_torch::ExecutionModeGuard guard(0);
    check(!guard.changed_inference && !guard.changed_grad, "default mode should not rewrite TLS");
  }
  {
    sputnik_torch::ExecutionModeGuard outer(2);
    sputnik_torch::ExecutionModeGuard inner(2);
    check(!inner.changed_inference && !inner.changed_grad, "matching inference scope should reuse TLS");
  }
#endif
  check(original.matches(), "test fails to restore original state");
}
} // namespace
int main() {
  try {
    verify();
    std::exception_ptr error;
    std::thread worker([&] { try { verify(); } catch (...) { error = std::current_exception(); } });
    worker.join();
    if (error) std::rethrow_exception(error);
    std::cout << "PASS execution mode equivalence, nesting, faults and carrier threads\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
