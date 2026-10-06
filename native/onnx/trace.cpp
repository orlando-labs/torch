#include "lowering.h"
#define ONNX_NAMESPACE onnx
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/serialization/export.h>
#include <atomic>
#include <algorithm>
#include <mutex>
#include <unordered_set>

namespace sputniktorch {
namespace {
using namespace onnx_export;
struct Session {
  int64_t token;
  std::mutex mutex;
  std::shared_ptr<torch::jit::tracer::TracingState> tracing =
      std::make_shared<torch::jit::tracer::TracingState>();
  std::vector<at::Tensor> inputs;
  std::map<std::string, at::Tensor> initializers;
  bool finished = false;
};
struct TraceCore { std::shared_ptr<Session> session; };
struct CurrentTrace {
  std::shared_ptr<torch::jit::tracer::TracingState> previous = torch::jit::tracer::getTracingState();
  explicit CurrentTrace(const std::shared_ptr<torch::jit::tracer::TracingState> &state) {
    torch::jit::tracer::setTracingState(state);
  }
  ~CurrentTrace() { torch::jit::tracer::setTracingState(std::move(previous)); }
};
std::atomic<int64_t> next_token{1};
std::mutex sessions_mutex;
std::unordered_map<int64_t, std::weak_ptr<Session>> sessions;
constexpr char trace_tag[] = "sputniktorch.ONNXTraceCore";

std::shared_ptr<Session> find_session(int64_t token) {
  std::lock_guard lock(sessions_mutex);
  auto found = sessions.find(token);
  auto session = found == sessions.end() ? nullptr : found->second.lock();
  require(session != nullptr, error_kind, "ONNX trace context has expired");
  return session;
}

TraceCore &trace_core(SputnikCtx *cx, SputnikValue value) {
  void *ptr = nullptr;
  if (!sputnik_handle_ptr(cx, value, trace_tag, &ptr)) throw FaultPending{};
  return *static_cast<TraceCore *>(ptr);
}

std::vector<std::string> strings(SputnikCtx *cx, SputnikValue value) {
  require(sputnik_is_list(cx, value), error_kind, "ONNX names must be an Array");
  std::vector<std::string> names;
  for (size_t i = 0; i < sputnik_list_len(cx, value); ++i) names.push_back(string(cx, sputnik_list_at(cx, value, i)));
  return names;
}

void name_values(const std::vector<std::string> &names, at::ArrayRef<Value *> values,
                 std::unordered_set<std::string> &used) {
  require(names.size() == values.size(), error_kind, "ONNX name count does not match graph");
  for (size_t i = 0; i < names.size(); ++i) {
    require(!names[i].empty() && names[i].find('\0') == std::string::npos &&
                !(names[i][0] >= '0' && names[i][0] <= '9'),
            error_kind, "ONNX names must be nonempty and must not start with a digit");
    require(used.insert(names[i]).second, error_kind, "duplicate ONNX value name: " + names[i]);
    values[i]->setDebugName(names[i]);
    require(values[i]->debugName() == names[i], error_kind, "ONNX name collides with another graph value: " + names[i]);
  }
}
}

struct TraceKernelGuard::Impl {
  std::shared_ptr<Session> session;
  std::unique_lock<std::mutex> lock;
  std::shared_ptr<torch::jit::tracer::TracingState> previous;
  explicit Impl(int64_t token) : session(find_session(token)), lock(session->mutex),
      previous(torch::jit::tracer::getTracingState()) {
    require(!session->finished, onnx_export::error_kind, "ONNX trace is already finished");
    torch::jit::tracer::setTracingState(session->tracing);
  }
  ~Impl() { torch::jit::tracer::setTracingState(std::move(previous)); }
};
TraceKernelGuard::TraceKernelGuard(int64_t token) {
  if (token > 0) impl_ = std::make_unique<Impl>(token);
}
TraceKernelGuard::~TraceKernelGuard() = default;

extern "C" void amt_onnx_trace_free(SputnikCtx *, void *ptr) {
  auto core = static_cast<TraceCore *>(ptr);
  {
    std::lock_guard lock(sessions_mutex);
    sessions.erase(core->session->token);
  }
  delete core;
}

extern "C" SputnikStatus amt_onnx_trace_open(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 2);
    at::NoGradGuard no_grad;
    require(sputnik_is_list(cx, a[0]) && sputnik_is_list(cx, a[1]), error_kind, "invalid ONNX trace inputs");
    auto session = std::make_shared<Session>();
    session->token = next_token.fetch_add(1);
    require(session->token > 0 && session->token <= (std::numeric_limits<int64_t>::max() - 2) / 4,
            error_kind, "ONNX trace token exhausted");
    session->tracing->warn = false;
    std::unordered_set<const c10::TensorImpl *> tensors;
    auto bind = [&](const at::Tensor &tensor, const std::string &name) {
      tensor_dtype(tensor.scalar_type());
      auto input = session->tracing->graph->addInput(name);
      input->setType(c10::TensorType::create(tensor));
      session->tracing->setValue(tensor, input);
    };
    for (size_t i = 0; i < sputnik_list_len(cx, a[0]); ++i) {
      auto tensor = core(cx, sputnik_list_at(cx, a[0], i)).tensor;
      require(tensors.insert(tensor.unsafeGetTensorImpl()).second, error_kind,
              "ONNX example inputs must have distinct tensor identities");
      session->inputs.push_back(tensor);
      bind(tensor, "input_" + std::to_string(i));
    }
    for (size_t i = 0; i < sputnik_list_len(cx, a[1]); ++i) {
      auto pair = sputnik_list_at(cx, a[1], i);
      require(sputnik_is_list(cx, pair) && sputnik_list_len(cx, pair) == 2, error_kind, "invalid model state entry");
      auto name = "state." + string(cx, sputnik_list_at(cx, pair, 0));
      auto tensor = core(cx, sputnik_list_at(cx, pair, 1)).tensor;
      if (!tensors.insert(tensor.unsafeGetTensorImpl()).second) {
        require(std::none_of(session->inputs.begin(), session->inputs.end(), [&](const at::Tensor &input) {
                  return input.unsafeGetTensorImpl() == tensor.unsafeGetTensorImpl();
                }),
                error_kind, "ONNX inputs cannot alias model state");
        continue;
      }
      bind(tensor, name);
      session->initializers.emplace(name, tensor.detach().cpu().contiguous().clone());
    }
    auto handle = std::make_unique<TraceCore>(TraceCore{session});
    *out = sputnik_make_handle(cx, trace_tag, handle.get());
    if (!*out) throw FaultPending{};
    {
      std::lock_guard lock(sessions_mutex);
      sessions.emplace(session->token, session);
    }
    handle.release();
  });
}

extern "C" SputnikStatus amt_onnx_trace_token(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 1);
    *out = sputnik_make_int(cx, trace_core(cx, a[0]).session->token);
  });
}

extern "C" SputnikStatus amt_onnx_trace_finish(SputnikCtx *cx, const SputnikValue *a, size_t n, SputnikValue *out) {
  return boundary(cx, [&] {
    argc_check(n, 3);
    auto session = trace_core(cx, a[0]).session;
    std::lock_guard lock(session->mutex);
    require(!session->finished, error_kind, "ONNX trace is already finished");
    auto options = a[2];
    require(sputnik_is_list(cx, options) && sputnik_list_len(cx, options) == 4, error_kind, "invalid ONNX options");
    const auto opset = integer(cx, sputnik_list_at(cx, options, 0));
    require(opset >= 18 && opset <= 20, error_kind, "supported ONNX opset versions are 18 through 20");
    {
      CurrentTrace tracing(session->tracing);
      for (size_t i = 0; i < sputnik_list_len(cx, a[1]); ++i) {
        auto tensor = core(cx, sputnik_list_at(cx, a[1], i)).tensor;
        require(session->tracing->hasValue(tensor), error_kind, "output has no traced tensor dataflow");
        session->tracing->graph->registerOutput(session->tracing->getValue(tensor));
      }
    }
    auto graph = lower_graph(session->tracing->graph, opset);
    std::unordered_set<std::string> used;
    for (const auto &[name, tensor] : session->initializers) used.insert(name);
    auto input_names = strings(cx, sputnik_list_at(cx, options, 1));
    name_values(input_names, graph->inputs().slice(0, session->inputs.size()), used);
    name_values(strings(cx, sputnik_list_at(cx, options, 2)), graph->outputs(), used);
    std::unordered_map<std::string, std::unordered_map<int64_t, std::string>> dynamic;
    auto axes = sputnik_list_at(cx, options, 3);
    for (size_t i = 0; i < sputnik_list_len(cx, axes); ++i) {
      auto entry = sputnik_list_at(cx, axes, i);
      auto name = string(cx, sputnik_list_at(cx, entry, 0));
      auto found = std::find_if(graph->inputs().begin(), graph->inputs().begin() + session->inputs.size(),
                                [&](Value *v) { return v->debugName() == name; });
      Value *value = found == graph->inputs().begin() + session->inputs.size() ? nullptr : *found;
      for (auto output : graph->outputs()) if (output->debugName() == name) value = output;
      require(value != nullptr, error_kind, "dynamic_axes contains unknown input/output: " + name);
      auto dimensions = sputnik_list_at(cx, entry, 1);
      for (size_t j = 0; j < sputnik_list_len(cx, dimensions); ++j) {
        auto pair = sputnik_list_at(cx, dimensions, j);
        auto axis = integer(cx, sputnik_list_at(cx, pair, 0));
        auto label = string(cx, sputnik_list_at(cx, pair, 1));
        require(axis >= 0 && axis < tensor_rank(value) && !label.empty(), error_kind, "invalid dynamic ONNX axis");
        dynamic[name][axis] = label;
      }
    }
    auto model = torch::jit::export_onnx(graph, session->initializers, opset, dynamic, false,
        torch::onnx::OperatorExportTypes::ONNX, true, false);
    auto bytes = torch::jit::serialize_model_proto_to_string(std::get<0>(model));
    torch::jit::check_onnx_proto(bytes);
    std::vector<SputnikValue> encoded{
      sputnik_make_bytes(cx, reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()),
      sputnik_make_int(cx, std::distance(graph->nodes().begin(), graph->nodes().end())),
      sputnik_make_int(cx, static_cast<int64_t>(session->initializers.size()))
    };
    *out = sputnik_make_list(cx, encoded.data(), encoded.size());
    session->finished = true;
  });
}
}
