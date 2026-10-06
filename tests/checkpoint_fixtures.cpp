#include <caffe2/serialize/inline_container.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <torch/serialize.h>
#include <vector>

namespace {
void integer(std::string &p, int32_t v) {
  p += 'J';
  for (int i = 0; i < 4; ++i)
    p += char(uint32_t(v) >> (i * 8));
}
void string(std::string &p, const std::string &s) {
  p += 'X';
  for (int i = 0; i < 4; ++i)
    p += char(uint32_t(s.size()) >> (i * 8));
  p += s;
}
void tuple(std::string &p, std::initializer_list<int32_t> values) {
  p += '(';
  for (auto v : values)
    integer(p, v);
  p += 't';
}
void tensor(std::string &p, const std::string &name, int offset = 0) {
  string(p, name);
  p += "ctorch._utils\n_rebuild_tensor_v2\n((";
  string(p, "storage");
  p += "ctorch\nFloatStorage\n";
  string(p, "0");
  string(p, "cuda:3");
  integer(p, 6);
  p += "tQ";
  integer(p, offset);
  tuple(p, {2, 2});
  tuple(p, {3, 1});
  p += char(0x89);
  p += "ccollections\nOrderedDict\n)RtR";
}
void write(const std::filesystem::path &path, const std::string &pickle) {
  caffe2::serialize::PyTorchStreamWriter w(path.string());
  w.writeRecord("data.pkl", pickle.data(), pickle.size());
  w.writeRecord("byteorder", "little", 6);
  const float values[] = {1, 2, 3, 4, 5, 6};
  w.writeRecord("data/0", values, sizeof(values));
  w.writeEndOfFile();
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "verify") {
      std::ifstream f(argv[2], std::ios::binary);
      std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
      auto value = torch::pickle_load(bytes).toGenericDict();
      auto weight = value.at("weight").toTensor();
      TORCH_CHECK(weight.numel() == 6 && weight.sum().item<float>() == 21.f);
      std::cout << "PASS LibTorch reads sputniktorch state dict\n";
      return 0;
    }
    TORCH_CHECK(argc == 2, "usage: checkpoint_fixtures directory | verify file");
    std::filesystem::path dir(argv[1]);
    std::filesystem::create_directories(dir);
    std::string good = "\x80\x02}(";
    tensor(good, "left");
    tensor(good, "right", 1);
    good += "u.";
    write(dir / "cuda_views.pt", good);
    std::string ordered = "\x80\x02" "ccollections\nOrderedDict\n)R(";
    tensor(ordered, "left");
    ordered += "u}";
    string(ordered, "_metadata");
    ordered += '}';
    string(ordered, "");
    ordered += '}';
    string(ordered, "version");
    integer(ordered, 1);
    ordered += "sssb.";
    write(dir / "ordered_metadata.pt", ordered);
    std::string framed = "\x80\x04\x95";
    const uint64_t frame_size = good.size() - 2;
    for (int i = 0; i < 8; ++i)
      framed += char(frame_size >> (8 * i));
    framed += good.substr(2);
    write(dir / "protocol4.pt", framed);
    std::string bad_bounds = "\x80\x02}(";
    tensor(bad_bounds, "x", 100);
    bad_bounds += "u.";
    write(dir / "bad_bounds.pt", bad_bounds);
    write(dir / "forbidden_global.pt", "\x80\x02"
                                       "cos\nsystem\n.");
    write(dir / "truncated.pt", good.substr(0, good.size() - 1));
    auto unsupported = good;
    unsupported[1] = 5;
    write(dir / "unsupported.pt", unsupported);
    std::string duplicate = "\x80\x02}(";
    tensor(duplicate, "same");
    tensor(duplicate, "same");
    duplicate += "u.";
    write(dir / "duplicate.pt", duplicate);
    std::cout << "PASS checkpoint fixtures generated without an interpreter\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
