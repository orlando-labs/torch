#include "bridge.h"
#include "lowering.h"
#include <atomic>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>

using namespace sputniktorch;
namespace {
std::atomic<uint64_t> next_file{0};
struct TemporaryFile {
  int fd = -1;
  std::string path;
  bool owned = false;
  ~TemporaryFile() {
    if (fd >= 0) ::close(fd);
    if (owned && !path.empty()) ::unlink(path.c_str());
  }
};
}

extern "C" SputnikStatus amt_onnx_save(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 2);
    const uint8_t *bytes;
    size_t length;
    require(sputnik_bytes_view(cx, a[0], &bytes, &length), "TypeError", "ONNX program must contain Bytes");
    auto path = string(cx, a[1]);
    require(!path.empty() && path.find('\0') == std::string::npos, onnx_export::error_kind, "invalid ONNX path");
    TemporaryFile file;
    auto fail = [&] { throw Error(onnx_export::error_kind, "cannot save ONNX program: " + std::string(std::strerror(errno))); };
    for (int attempt = 0; attempt < 32; ++attempt) {
      file.path = path + ".part." + std::to_string(::getpid()) + "." + std::to_string(next_file.fetch_add(1));
      file.fd = ::open(file.path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
      if (file.fd >= 0) { file.owned = true; break; }
      if (errno != EEXIST) fail();
      file.path.clear();
    }
    if (file.fd < 0) fail();
    for (size_t offset = 0; offset < length;) {
      auto written = ::write(file.fd, bytes + offset, std::min(length - offset, static_cast<size_t>(1 << 20)));
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) fail();
      offset += static_cast<size_t>(written);
    }
    if (::fsync(file.fd) != 0) fail();
    auto status = ::close(file.fd);
    file.fd = -1;
    if (status != 0 || ::rename(file.path.c_str(), path.c_str()) != 0) fail();
    file.path.clear();
    *out = sputnik_make_null(cx);
  });
}
