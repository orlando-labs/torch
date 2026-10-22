#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>
extern char **environ;
namespace {
volatile sig_atomic_t interrupted = 0;
void interrupt(int sig) {
  interrupted = sig;
}
} // namespace
int main(int argc, char **argv) {
  std::vector<pid_t> children;
  std::string directory;
  try {
    if (argc < 4 || std::string(argv[1]) != "--nproc") {
      std::cerr << "usage: torch-run --nproc N [--timeout SECONDS] program [args...]\n";
      return 2;
    }
    const int count = std::stoi(argv[2]);
    int timeout = 60, start = 3;
    if (argc > 5 && std::string(argv[3]) == "--timeout") {
      timeout = std::stoi(argv[4]);
      start = 5;
    }
    if (count < 1 || count > 64 || timeout < 1 || timeout > 86400 || start >= argc)
      throw std::runtime_error("invalid worker count, timeout, or program");
    char path[] = "/tmp/torch-run-XXXXXX";
    if (!mkdtemp(path))
      throw std::runtime_error("cannot create rendezvous directory");
    directory = path;
    std::signal(SIGINT, interrupt);
    std::signal(SIGTERM, interrupt);
    for (int rank = 0; rank < count; ++rank) {
      std::map<std::string, std::string> env;
      for (char **p = environ; p && *p; ++p) {
        std::string item(*p);
        auto at = item.find('=');
        if (at != std::string::npos)
          env[item.substr(0, at)] = item.substr(at + 1);
      }
      env["RANK"] = std::to_string(rank);
      env["LOCAL_RANK"] = std::to_string(rank);
      env["WORLD_SIZE"] = std::to_string(count);
      env["SPUTNIK_RENDEZVOUS"] = directory + "/store";
      std::vector<std::string> strings;
      for (const auto &[key, value] : env)
        strings.push_back(key + "=" + value);
      std::vector<char *> values;
      for (auto &s : strings)
        values.push_back(s.data());
      values.push_back(nullptr);
      posix_spawnattr_t attributes;
      posix_spawnattr_init(&attributes);
      posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
      posix_spawnattr_setpgroup(&attributes, 0);
      pid_t pid;
      int rc = posix_spawnp(&pid, argv[start], nullptr, &attributes, argv + start, values.data());
      posix_spawnattr_destroy(&attributes);
      if (rc)
        throw std::runtime_error(std::string("worker spawn failed: ") + std::strerror(rc));
      children.push_back(pid);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout);
    bool running = true;
    while (running) {
      if (interrupted)
        throw std::runtime_error("launcher interrupted");
      if (std::chrono::steady_clock::now() >= deadline)
        throw std::runtime_error("worker group timeout");
      running = false;
      for (auto &pid : children) {
        if (pid <= 0)
          continue;
        int status = 0;
        auto result = waitpid(pid, &status, WNOHANG);
        if (result == 0) {
          running = true;
          continue;
        }
        if (result < 0) {
          if (errno == EINTR) {
            running = true;
            continue;
          }
          throw std::runtime_error("waitpid failed");
        }
        pid = 0;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
          throw std::runtime_error("worker failed; terminating peers");
      }
      if (running)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::filesystem::remove_all(directory);
    return 0;
  } catch (const std::exception &e) {
    // Each worker has its own process group so descendants are cleaned up too.
    for (auto pid : children)
      if (pid > 0)
        kill(-pid, SIGTERM);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    for (auto pid : children)
      if (pid > 0) {
        kill(-pid, SIGKILL);
        while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
        }
      }
    if (!directory.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(directory, ec);
    }
    std::cerr << "torch-run: " << e.what() << '\n';
    return interrupted ? 128 + interrupted : 1;
  }
}
