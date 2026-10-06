#include "bridge.h"
#include <caffe2/serialize/inline_container.h>
#include <cstring>
#include <filesystem>
#include <torch/mps.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

// Non-executing protocol-2/4 parser for a flat state_dict. GLOBAL and REDUCE
// are data tokens handled by this allowlist, never passed to a Python/JIT VM.
namespace {
using namespace sputniktorch;
constexpr size_t max_pickle = 16 * 1024 * 1024, max_nodes = 1000000, max_records = 100000;
constexpr uint64_t max_bytes = 2ULL * 1024 * 1024 * 1024;
void valid(bool yes, const std::string &why) {
  require(yes, "Sputniktorch.Errors.CheckpointError", why);
}
at::ScalarType storage_type(const std::string &s) {
  static const std::unordered_map<std::string, at::ScalarType> types = {
      {"FloatStorage", at::kFloat}, {"DoubleStorage", at::kDouble},
      {"HalfStorage", at::kHalf},   {"BFloat16Storage", at::kBFloat16},
      {"LongStorage", at::kLong},   {"IntStorage", at::kInt},
      {"ShortStorage", at::kShort}, {"CharStorage", at::kChar},
      {"ByteStorage", at::kByte},   {"BoolStorage", at::kBool}};
  auto it = types.find(s);
  valid(it != types.end(), "unsupported storage type: " + s);
  return it->second;
}
std::string storage_name(at::ScalarType t) {
  switch (t) {
  case at::kFloat:
    return "FloatStorage";
  case at::kDouble:
    return "DoubleStorage";
  case at::kHalf:
    return "HalfStorage";
  case at::kBFloat16:
    return "BFloat16Storage";
  case at::kLong:
    return "LongStorage";
  case at::kInt:
    return "IntStorage";
  case at::kShort:
    return "ShortStorage";
  case at::kChar:
    return "CharStorage";
  case at::kByte:
    return "ByteStorage";
  case at::kBool:
    return "BoolStorage";
  default:
    throw Error("Sputniktorch.Errors.CheckpointError", "unsupported checkpoint dtype");
  }
}
struct Node {
  enum Kind { Nil, Int, Bool, Str, Tuple, Dict, Global, Storage, Tensor } kind = Nil;
  int64_t value = 0;
  std::string text;
  std::vector<size_t> children;
};
struct Parser {
  std::string bytes;
  size_t pos = 0;
  std::vector<Node> nodes;
  std::vector<size_t> stack, marks;
  std::unordered_map<uint32_t, size_t> memo;
  size_t metadata_visits = 0;
  size_t edges = 0;
  void add_edges(size_t count) {
    valid(count <= max_nodes - edges, "pickle container edge limit");
    edges += count;
  }
  explicit Parser(std::string b) : bytes(std::move(b)) {
    nodes.reserve(1024);
  }
  unsigned char byte() {
    valid(pos < bytes.size(), "truncated pickle");
    return bytes[pos++];
  }
  uint64_t uintn(size_t n) {
    valid(n <= 8 && n <= bytes.size() - pos, "truncated integer");
    uint64_t x = 0;
    for (size_t i = 0; i < n; ++i)
      x |= uint64_t(byte()) << (8 * i);
    return x;
  }
  std::string str(size_t n) {
    valid(n <= max_pickle && n <= bytes.size() - pos, "invalid string size");
    auto r = bytes.substr(pos, n);
    pos += n;
    return r;
  }
  std::string line() {
    auto end = bytes.find('\n', pos);
    valid(end != std::string::npos, "truncated GLOBAL");
    auto n = end - pos;
    return str(n + 1).substr(0, n);
  }
  size_t push(Node n) {
    valid(nodes.size() < max_nodes && stack.size() < max_nodes, "pickle node limit");
    add_edges(n.children.size());
    nodes.push_back(std::move(n));
    stack.push_back(nodes.size() - 1);
    return stack.back();
  }
  size_t pop() {
    valid(!stack.empty(), "pickle stack underflow");
    auto v = stack.back();
    stack.pop_back();
    return v;
  }
  Node &node(size_t i, Node::Kind k) {
    valid(i < nodes.size() && nodes[i].kind == k, "unexpected pickle value");
    return nodes[i];
  }
  std::string text(size_t i) {
    return node(i, Node::Str).text;
  }
  int64_t integer(size_t i) {
    return node(i, Node::Int).value;
  }
  std::vector<size_t> marked() {
    valid(!marks.empty(), "missing pickle MARK");
    auto start = marks.back();
    marks.pop_back();
    valid(start <= stack.size(), "invalid MARK");
    std::vector<size_t> a(stack.begin() + start, stack.end());
    stack.resize(start);
    return a;
  }
  void global(std::string module, std::string name) {
    if (module == "torch")
      storage_type(name);
    else
      valid((module == "collections" && name == "OrderedDict") ||
                (module == "torch._utils" &&
                 (name == "_rebuild_tensor_v2" || name == "_rebuild_tensor")),
            "forbidden pickle GLOBAL: " + module + "." + name);
    push({Node::Global, 0, module + "." + name, {}});
  }
  void pairs(size_t dict, const std::vector<size_t> &items) {
    valid(items.size() % 2 == 0, "invalid dictionary items");
    auto &d = node(dict, Node::Dict);
    valid(d.children.size() + items.size() < max_nodes, "dictionary limit");
    add_edges(items.size());
    d.children.insert(d.children.end(), items.begin(), items.end());
  }
  size_t parse() {
    valid(byte() == 0x80, "only binary pickle supported");
    auto protocol = byte();
    valid(protocol == 2 || protocol == 4, "supported pickle protocols: 2 and 4");
    while (pos < bytes.size()) {
      valid(stack.size() < max_nodes, "pickle stack limit");
      auto op = byte();
      switch (op) {
      case '.':
        valid(stack.size() == 1 && marks.empty() && pos == bytes.size(), "invalid STOP");
        return stack.back();
      case '(':
        valid(marks.size() < 128, "pickle nesting limit");
        marks.push_back(stack.size());
        break;
      case ')':
        push({Node::Tuple, 0, "", {}});
        break;
      case '}':
        push({Node::Dict, 0, "", {}});
        break;
      case 'N':
        push({});
        break;
      case 0x88:
      case 0x89:
        push({Node::Bool, op == 0x88, "", {}});
        break;
      case 'K':
        push({Node::Int, int64_t(uintn(1)), "", {}});
        break;
      case 'M':
        push({Node::Int, int64_t(uintn(2)), "", {}});
        break;
      case 'J':
        push({Node::Int, int32_t(uintn(4)), "", {}});
        break;
      case 0x8a: {
        auto n = byte();
        valid(n <= 8, "integer exceeds int64");
        auto v = uintn(n);
        if (n && n < 8 && (v >> (n * 8 - 1)))
          v |= (~uint64_t(0)) << (n * 8);
        push({Node::Int, int64_t(v), "", {}});
        break;
      }
      case 'X':
      case 0x8c: {
        auto n = uintn(op == 'X' ? 4 : 1);
        push({Node::Str, 0, str(n), {}});
        break;
      }
      case 'c': {
        auto m = line();
        auto n = line();
        global(m, n);
        break;
      }
      case 0x93: {
        auto n = text(pop());
        auto m = text(pop());
        global(m, n);
        break;
      }
      case 't': {
        auto a = marked();
        push({Node::Tuple, 0, "", std::move(a)});
        break;
      }
      case 0x85:
      case 0x86:
      case 0x87: {
        size_t n = op - 0x84;
        valid(stack.size() >= n, "tuple underflow");
        std::vector<size_t> a(stack.end() - n, stack.end());
        stack.resize(stack.size() - n);
        push({Node::Tuple, 0, "", std::move(a)});
        break;
      }
      case 'q':
      case 'r': {
        auto i = uintn(op == 'q' ? 1 : 4);
        valid(i < max_nodes && !stack.empty() && !memo.count(i), "invalid memo index");
        memo[i] = stack.back();
        break;
      }
      case 0x94:
        valid(!stack.empty() && memo.size() < max_nodes && !memo.count(memo.size()),
              "invalid MEMOIZE");
        memo[uint32_t(memo.size())] = stack.back();
        break;
      case 'h':
      case 'j': {
        auto i = uintn(op == 'h' ? 1 : 4);
        valid(memo.count(i), "missing memo");
        stack.push_back(memo[i]);
        break;
      }
      case 0x95: {
        auto length = uintn(8);
        valid(length <= bytes.size() - pos, "invalid FRAME");
        break;
      }
      case 's': {
        auto v = pop(), k = pop();
        valid(!stack.empty(), "SETITEM underflow");
        pairs(stack.back(), {k, v});
        break;
      }
      case 'u': {
        auto a = marked();
        valid(!stack.empty(), "SETITEMS underflow");
        pairs(stack.back(), a);
        break;
      }
      case 'Q': {
        auto args = node(pop(), Node::Tuple).children;
        valid(args.size() == 5 && text(args[0]) == "storage", "unsupported persistent id");
        auto name = node(args[1], Node::Global).text;
        valid(name.rfind("torch.", 0) == 0, "bad storage global");
        storage_type(name.substr(6));
        auto key = text(args[2]);
        valid(!key.empty() && key.find_first_not_of("0123456789") == std::string::npos,
              "invalid storage key");
        text(args[3]);
        valid(integer(args[4]) >= 0, "negative storage size");
        push({Node::Storage, 0, "", std::move(args)});
        break;
      }
      case 'R': {
        auto args = node(pop(), Node::Tuple).children;
        auto fn = node(pop(), Node::Global).text;
        if (fn == "collections.OrderedDict") {
          valid(args.empty(), "OrderedDict arguments unsupported");
          push({Node::Dict, 0, "", {}});
        } else {
          valid(fn == "torch._utils._rebuild_tensor_v2" || fn == "torch._utils._rebuild_tensor",
                "forbidden REDUCE");
          valid(args.size() == (fn.ends_with("v2") ? 6 : 4),
                "unsupported tensor rebuild signature");
          node(args[0], Node::Storage);
          integer(args[1]);
          node(args[2], Node::Tuple);
          node(args[3], Node::Tuple);
          if (args.size() == 6) {
            node(args[4], Node::Bool);
            valid(node(args[5], Node::Dict).children.empty(), "tensor hooks unsupported");
          }
          push({Node::Tensor, 0, "", std::move(args)});
        }
        break;
      }
      case 'b': {
        auto state = pop();
        valid(!stack.empty(), "BUILD underflow");
        node(stack.back(), Node::Dict);
        auto fields = node(state, Node::Dict).children;
        valid(fields.size() == 2 && text(fields[0]) == "_metadata",
              "only state_dict metadata BUILD is supported");
        validate_metadata(fields[1]);
        break;
      }
      default:
        throw Error("Sputniktorch.Errors.CheckpointError",
                    "unsupported pickle opcode " + std::to_string(op));
      }
    }
    throw Error("Sputniktorch.Errors.CheckpointError", "missing pickle STOP");
  }
  void validate_metadata(size_t id, size_t depth = 0) {
    valid(++metadata_visits <= max_nodes, "metadata traversal limit");
    valid(depth < 32, "cyclic/deep metadata");
    auto n = nodes.at(id);
    if (n.kind == Node::Int || n.kind == Node::Str || n.kind == Node::Bool || n.kind == Node::Nil)
      return;
    valid(n.kind == Node::Dict, "unsupported metadata");
    // Module metadata has string keys and small scalar dictionaries only.
    valid(n.children.size() < 100000, "metadata limit");
    for (size_t i = 0; i < n.children.size(); i += 2) {
      text(n.children[i]);
      validate_metadata(n.children[i + 1], depth + 1);
    }
  }
  std::vector<int64_t> dims(size_t id) {
    std::vector<int64_t> out;
    auto a = node(id, Node::Tuple).children;
    valid(a.size() <= 64, "rank limit");
    for (auto i : a) {
      auto v = integer(i);
      valid(v >= 0, "negative dimension/stride");
      out.push_back(v);
    }
    return out;
  }
};
struct Plan {
  std::string key, storage, source;
  at::ScalarType type;
  int64_t count, offset;
  std::vector<int64_t> shape, stride;
};
std::string read(caffe2::serialize::PyTorchStreamReader &r, const std::string &name, size_t limit) {
  auto n = r.getRecordSize(name);
  valid(n <= limit, "record exceeds limit: " + name);
  auto [data, size] = r.getRecord(name);
  return std::string(static_cast<char *>(data.get()), size);
}
at::Device target(const std::string &name) {
  valid(name == "cpu" || name == "cpu:0" || name == "mps" || name == "mps:0",
        "unsupported checkpoint destination: " + name);
  if (name.starts_with("mps")) {
    valid(torch::mps::is_available(), "MPS unavailable");
    return at::Device(at::kMPS);
  }
  return at::Device(at::kCPU);
}
std::string mapping(SputnikCtx *cx, SputnikValue value, const std::string &source) {
  if (sputnik_is_str(cx, value))
    return string(cx, value);
  valid(sputnik_is_list(cx, value), "map_location must be a device or mapping");
  for (size_t i = 0; i < sputnik_list_len(cx, value); ++i) {
    auto pair = sputnik_list_at(cx, value, i);
    valid(sputnik_is_list(cx, pair) && sputnik_list_len(cx, pair) == 2, "invalid device mapping");
    if (string(cx, sputnik_list_at(cx, pair, 0)) == source)
      return string(cx, sputnik_list_at(cx, pair, 1));
  }
  return source;
}
void put32(std::string &s, uint32_t i) {
  for (int n = 0; n < 4; ++n)
    s.push_back(char(i >> (8 * n)));
}
void putstr(std::string &s, const std::string &v) {
  s += 'X';
  put32(s, v.size());
  s += v;
}
void putint(std::string &s, int64_t v) {
  s += char(0x8a);
  s += char(8);
  for (int n = 0; n < 8; ++n)
    s.push_back(char(uint64_t(v) >> (n * 8)));
}
void puttuple(std::string &s, at::IntArrayRef v) {
  s += '(';
  for (auto i : v)
    putint(s, i);
  s += 't';
}
} // namespace
extern "C" {
SputnikStatus amt_load(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 2);
    caffe2::serialize::PyTorchStreamReader reader(string(cx, a[0]));
    auto records = reader.getAllRecords();
    valid(records.size() <= max_records, "too many ZIP records");
    std::unordered_set<std::string> unique;
    uint64_t total = 0;
    for (const auto &r : records) {
      valid(unique.insert(r).second, "duplicate ZIP record");
      auto size = reader.getRecordSize(r);
      valid(size <= max_bytes - total, "checkpoint exceeds 2 GiB decoded limit");
      total += size;
    }
    valid(reader.hasRecord("byteorder") && read(reader, "byteorder", 16) == "little",
          "checkpoint must declare little byteorder");
    Parser p(read(reader, "data.pkl", max_pickle));
    auto root = p.parse();
    auto items = p.node(root, Node::Dict).children;
    valid(items.size() / 2 <= 100000, "too many state dict entries");
    std::vector<Plan> plans;
    std::unordered_set<std::string> names;
    for (size_t i = 0; i < items.size(); i += 2) {
      Plan v;
      v.key = p.text(items[i]);
      valid(names.insert(v.key).second, "duplicate state dict key");
      auto t = p.node(items[i + 1], Node::Tensor).children;
      auto s = p.node(t[0], Node::Storage).children;
      v.storage = p.text(s[2]);
      v.source = p.text(s[3]);
      v.type = storage_type(p.node(s[1], Node::Global).text.substr(6));
      v.count = p.integer(s[4]);
      v.offset = p.integer(t[1]);
      v.shape = p.dims(t[2]);
      v.stride = p.dims(t[3]);
      valid(v.shape.size() == v.stride.size() && v.offset >= 0, "invalid tensor geometry");
      auto bytes = uint64_t(v.count);
      valid(bytes <= max_bytes / at::elementSize(v.type), "storage size limit");
      bytes *= at::elementSize(v.type);
      valid(reader.hasRecord("data/" + v.storage) &&
                reader.getRecordSize("data/" + v.storage) == bytes,
            "storage byte size mismatch");
      uint64_t last = v.offset, numel = 1;
      bool empty = false;
      for (size_t d = 0; d < v.shape.size(); ++d) {
        auto dim = uint64_t(v.shape[d]), stride = uint64_t(v.stride[d]);
        empty |= dim == 0;
        valid(dim == 0 || numel <= uint64_t(INT64_MAX) / dim, "tensor numel overflow");
        numel *= dim;
        if (dim > 0) {
          valid(stride == 0 || (dim - 1) <= (uint64_t(INT64_MAX) - last) / stride,
                "storage bounds overflow");
          last += (dim - 1) * stride;
        }
      }
      valid(empty ? uint64_t(v.offset) <= uint64_t(v.count) : last < uint64_t(v.count),
            "tensor exceeds storage bounds");
      plans.push_back(std::move(v));
    }
    // Validate all destinations and storage descriptors before allocating tensors.
    std::unordered_map<std::string, Plan> storages;
    std::unordered_map<std::string, at::Device> destinations;
    for (const auto &p : plans) {
      auto d = target(mapping(cx, a[1], p.source));
      valid(!(d.is_mps() && p.type == at::kDouble), "MPS does not support float64");
      auto [it, added] = storages.emplace(p.storage, p);
      if (!added)
        valid(it->second.type == p.type && it->second.count == p.count &&
                  it->second.source == p.source,
              "conflicting storage descriptor");
      destinations.emplace(p.storage, d);
    }
    std::unordered_map<std::string, at::Tensor> loaded;
    at::NoGradGuard guard;
    for (const auto &[key, p] : storages) {
      auto t = at::empty({p.count}, at::TensorOptions().dtype(p.type).device(at::kCPU));
      reader.getRecord("data/" + key, t.data_ptr(), t.nbytes());
      loaded.emplace(key, t.to(destinations.at(key)));
    }
    std::vector<SputnikValue> result;
    for (const auto &p : plans) {
      SputnikValue pair[] = {text(cx, p.key),
                           wrap(cx, loaded.at(p.storage).as_strided(p.shape, p.stride, p.offset))};
      result.push_back(sputnik_make_list(cx, pair, 2));
    }
    *out = sputnik_make_list(cx, result.data(), result.size());
  });
}
SputnikStatus amt_save(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 2);
    valid(sputnik_is_list(cx, a[0]), "expected state dict entries");
    auto path = string(cx, a[1]);
    valid(!path.empty(), "empty checkpoint path");
    std::vector<std::pair<std::string, at::Tensor>> entries;
    std::unordered_set<std::string> names;
    uint64_t bytes = 0;
    at::NoGradGuard guard;
    auto count = sputnik_list_len(cx, a[0]);
    valid(count <= 100000, "too many state dict entries");
    for (size_t i = 0; i < count; ++i) {
      auto pair = sputnik_list_at(cx, a[0], i);
      valid(sputnik_is_list(cx, pair) && sputnik_list_len(cx, pair) == 2, "invalid state dict entry");
      auto key = string(cx, sputnik_list_at(cx, pair, 0));
      valid(names.insert(key).second, "duplicate key");
      auto &t = core(cx, sputnik_list_at(cx, pair, 1)).tensor;
      storage_name(t.scalar_type());
      valid(t.layout() == at::kStrided, "only dense tensors supported");
      valid(t.nbytes() <= max_bytes - bytes, "checkpoint size limit");
      bytes += t.nbytes();
      entries.emplace_back(key, t.detach().cpu().contiguous());
    }
    std::string pickle = "\x80\x02}(";
    for (size_t i = 0; i < entries.size(); ++i) {
      auto &[key, t] = entries[i];
      putstr(pickle, key);
      pickle += "ctorch._utils\n_rebuild_tensor_v2\n((";
      putstr(pickle, "storage");
      pickle += "ctorch\n" + storage_name(t.scalar_type()) + "\n";
      putstr(pickle, std::to_string(i));
      putstr(pickle, "cpu");
      putint(pickle, t.numel());
      pickle += "tQ";
      putint(pickle, 0);
      puttuple(pickle, t.sizes());
      puttuple(pickle, t.strides());
      pickle += char(0x89);
      pickle += "ccollections\nOrderedDict\n)RtR";
      valid(pickle.size() <= max_pickle, "pickle size limit");
    }
    pickle += "u.";
    std::string tmp = path + ".tmp.XXXXXX";
    std::vector<char> name(tmp.begin(), tmp.end());
    name.push_back(0);
    auto fd = mkstemp(name.data());
    valid(fd >= 0, "cannot create checkpoint temporary file");
    close(fd);
    tmp = name.data();
    try {
      {
        caffe2::serialize::PyTorchStreamWriter writer(tmp);
        writer.writeRecord("data.pkl", pickle.data(), pickle.size());
        writer.writeRecord("byteorder", "little", 6);
        for (size_t i = 0; i < entries.size(); ++i) {
          auto &t = entries[i].second;
          writer.writeRecord("data/" + std::to_string(i), t.data_ptr(), t.nbytes());
        }
        writer.writeEndOfFile();
      }
      std::filesystem::rename(tmp, path);
    } catch (...) {
      std::error_code ec;
      std::filesystem::remove(tmp, ec);
      throw;
    }
    *out = sputnik_make_null(cx);
  });
}
}
