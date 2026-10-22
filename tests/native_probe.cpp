#include <iostream>
#include <stdexcept>
#include <torch/mps.h>
#include "bridge.h"

int main(int argc, char **argv) {
  try {
    const auto device = sputnik_torch::tensor_device(argc > 1 ? argv[1] : "cpu");
    if (device.is_mps() && !torch::mps::is_available())
      throw std::runtime_error("MPS is not available to this process");
    auto x =
        torch::tensor({1.f, 2.f, 3.f}, torch::TensorOptions().device(device).requires_grad(true));
    auto loss = x.square().sum();
    loss.backward();
    TORCH_CHECK(loss.item<float>() == 14.f);
    TORCH_CHECK(torch::equal(x.grad().cpu(), torch::tensor({2.f, 4.f, 6.f})));
    auto w = torch::zeros({1}, torch::TensorOptions().device(device).requires_grad(true));
    for (int i = 0; i < 60; ++i) {
      auto error = (w * 2 - 6).square().mean();
      error.backward();
      torch::NoGradGuard guard;
      w.add_(w.grad(), -0.05);
      w.mutable_grad() = torch::Tensor();
    }
    TORCH_CHECK(std::abs(w.item<float>() - 3.f) < 1e-4);
    sputnik_torch::synchronize_devices();
    std::cout << "PASS " << device
              << " forward=14 gradient=[2,4,6] trained_weight=" << w.item<float>() << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
