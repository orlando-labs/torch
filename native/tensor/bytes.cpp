#include "bridge.h"
#include <cstring>

using namespace sputniktorch;

extern "C" SputnikStatus amt_from_bytes(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 5);
    ExecutionGuard mode(integer(cx, a[4]));
    const uint8_t *data = nullptr;
    size_t size = 0;
    require(sputnik_bytes_view(cx, a[0], &data, &size), "TypeError", "from_bytes expects Bytes");
    const auto offset = integer(cx, a[1]);
    require(offset >= 0 && static_cast<uint64_t>(offset) <= size, "ArgumentError",
            "byte offset is out of bounds");
    const auto length = size - static_cast<size_t>(offset);
    require(length <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
            "Sputniktorch.Errors.ShapeError", "byte payload exceeds int64");
    auto shape = sputnik_is_null(cx, a[2]) ? std::vector<int64_t>{static_cast<int64_t>(length)}
                                       : integers(cx, a[2]);
    require(shape.size() <= 64, "Sputniktorch.Errors.ShapeError", "rank exceeds 64");
    int64_t count = 1;
    for (const auto dimension : shape) {
      require(dimension >= 0, "Sputniktorch.Errors.ShapeError", "negative byte tensor dimension");
      require(dimension == 0 || count <= std::numeric_limits<int64_t>::max() / dimension,
              "Sputniktorch.Errors.ShapeError", "byte tensor shape overflows int64");
      count *= dimension;
    }
    require(static_cast<uint64_t>(count) == length, "Sputniktorch.Errors.ShapeError",
            "shape must consume the entire byte payload after offset");
    const auto name = string(cx, a[3]);
    require(name == "cpu" || name == "cpu:0" || name == "mps" || name == "mps:0",
            "Sputniktorch.Errors.DeviceError", "from_bytes supports cpu and mps");
    auto tensor = at::empty(shape, at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    if (length != 0)
      std::memcpy(tensor.data_ptr<uint8_t>(), data + offset, length);
    const auto placement = name.starts_with("mps") ? at::Device(at::kMPS) : at::Device(at::kCPU);
    *out = wrap(cx, tensor.to(placement));
  });
}
