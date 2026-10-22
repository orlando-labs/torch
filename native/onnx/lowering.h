#pragma once
#include "bridge.h"
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/constants.h>
#include <map>
#include <unordered_map>

namespace sputnik_torch::onnx_export {
using torch::jit::Graph;
using torch::jit::Node;
using torch::jit::Value;

inline constexpr char error_kind[] = "Torch.Errors.ONNXError";

int64_t tensor_dtype(at::ScalarType dtype);
at::ScalarType scalar_type(Value *value);
int64_t tensor_rank(Value *value);
std::vector<int64_t> tensor_shape(Value *value);
c10::IValue literal(Value *value);
bool absent(Value *value);
int64_t integer_value(Value *value);
double number_value(Value *value);
bool boolean_value(Value *value);
std::vector<int64_t> integer_list(Value *value);

struct Lowering {
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  std::unordered_map<Value *, Value *> values;
  int64_t opset;

  explicit Lowering(int64_t opset) : opset(opset) {}
  Value *input(Value *value);
  Value *constant(const at::Tensor &tensor);
  Value *scalar(double value, at::ScalarType dtype);
  Value *ints(const std::vector<int64_t> &values);
  Value *cast(Value *value, at::ScalarType dtype);
  Value *typed(Value *value, at::ScalarType dtype);
  Node *node(const std::string &op, const std::vector<Value *> &inputs);
  Value *op(const std::string &op, const std::vector<Value *> &inputs);
  void result(Node *source, Value *target);
  bool lower_tensor(Node *node);
  bool lower_nn(Node *node);
};

std::shared_ptr<Graph> lower_graph(const std::shared_ptr<Graph> &source, int64_t opset);
}
