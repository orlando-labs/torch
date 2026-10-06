#include <iostream>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/torch.h>

int main(int argc, char **argv) {
  try {
    TORCH_CHECK(argc == 3, "usage: probe rank rendezvous-file");
    const int rank = std::stoi(argv[1]);
    const auto timeout = std::chrono::seconds(20);
    auto store = c10::make_intrusive<c10d::FileStore>(argv[2], 2);
    store->setTimeout(timeout);
    auto options = c10d::ProcessGroupGloo::Options::create(timeout);
    options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));
    auto group = c10::make_intrusive<c10d::ProcessGroupGloo>(store, rank, 2, options);
    std::vector<at::Tensor> values{torch::tensor({float(rank + 1)})};
    group->allreduce(values)->wait(timeout);
    TORCH_CHECK(values[0].item<float>() == 3.f, "allreduce sum");
    values[0].fill_(rank == 0 ? 42.f : 0.f);
    group->broadcast(values)->wait(timeout);
    TORCH_CHECK(values[0].item<float>() == 42.f, "broadcast");
    group->barrier()->wait(timeout);
    std::cout << "PASS Gloo rank=" << rank << " sum=3 broadcast=42\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
