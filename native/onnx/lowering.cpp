#include "lowering.h"
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace sputnik_torch::onnx_export {
int64_t tensor_dtype(at::ScalarType dtype) {
  switch (dtype) {
  case at::kFloat: return 1;
  case at::kByte: return 2;
  case at::kChar: return 3;
  case at::kShort: return 5;
  case at::kInt: return 6;
  case at::kLong: return 7;
  case at::kBool: return 9;
  case at::kHalf: return 10;
  case at::kDouble: return 11;
  case at::kBFloat16: return 16;
  default: throw Error(error_kind, "unsupported ONNX tensor dtype");
  }
}

at::ScalarType scalar_type(Value *value) {
  auto type = value->type()->cast<c10::TensorType>();
  require(type && type->scalarType(), error_kind, "ONNX requires a known tensor dtype");
  return *type->scalarType();
}

int64_t tensor_rank(Value *value) {
  auto type = value->type()->cast<c10::TensorType>();
  require(type && type->dim(), error_kind, "ONNX requires a known tensor rank");
  return static_cast<int64_t>(*type->dim());
}

std::vector<int64_t> tensor_shape(Value *value) {
  auto type = value->type()->cast<c10::TensorType>();
  auto shape = type ? type->sizes().concrete_sizes() : std::nullopt;
  require(shape.has_value(), error_kind, "operator requires a known example shape");
  return *shape;
}

c10::IValue literal(Value *value) {
  if (auto constant = torch::jit::toIValue(value))
    return *constant;
  if (value->node()->kind() == c10::prim::ListConstruct) {
    c10::List<int64_t> items;
    for (auto item : value->node()->inputs())
      items.push_back(integer_value(item));
    return items;
  }
  throw Error(error_kind, "operator argument must be constant: %" + value->debugName());
}

bool absent(Value *value) { return value->type()->kind() == c10::TypeKind::NoneType; }
int64_t integer_value(Value *value) { return literal(value).toInt(); }
double number_value(Value *value) {
  auto item = literal(value);
  return item.isInt() ? static_cast<double>(item.toInt()) : item.toDouble();
}
bool boolean_value(Value *value) { return literal(value).toBool(); }
std::vector<int64_t> integer_list(Value *value) { return literal(value).toIntVector(); }

Value *Lowering::constant(const at::Tensor &tensor) {
  auto value = tensor.detach().cpu().contiguous();
  auto node = graph->create(c10::Symbol::fromQualString("onnx::Constant"), 1);
  node->t_(c10::attr::value, value);
  node->output()->setType(c10::TensorType::create(value));
  graph->insertNode(node);
  return node->output();
}

Value *Lowering::scalar(double value, at::ScalarType dtype) {
  return constant(at::scalar_tensor(value, at::TensorOptions().dtype(dtype).device(at::kCPU)));
}

Value *Lowering::ints(const std::vector<int64_t> &items) {
  return constant(at::tensor(items, at::TensorOptions().dtype(at::kLong)));
}

Value *Lowering::input(Value *value) {
  if (auto found = values.find(value); found != values.end())
    return found->second;
  auto item = literal(value);
  require(item.isTensor(), error_kind, "expected a tensor input: %" + value->debugName());
  auto result = constant(item.toTensor());
  values[value] = result;
  return result;
}

Value *Lowering::cast(Value *value, at::ScalarType dtype) {
  auto type = value->type()->cast<c10::TensorType>();
  if (type && type->scalarType() == dtype)
    return value;
  auto converted = node("Cast", {value});
  converted->i_(c10::Symbol::attr("to"), tensor_dtype(dtype));
  if (type)
    converted->output()->setType(type->withScalarType(dtype));
  return converted->output();
}

Value *Lowering::typed(Value *value, at::ScalarType dtype) {
  if (value->type()->cast<c10::TensorType>())
    return cast(input(value), dtype);
  auto item = literal(value);
  if (item.isBool())
    return scalar(item.toBool(), dtype);
  if (item.isInt())
    return constant(at::scalar_tensor(item.toInt(), at::TensorOptions().dtype(dtype)));
  return scalar(number_value(value), dtype);
}

Node *Lowering::node(const std::string &op, const std::vector<Value *> &inputs) {
  auto result = graph->create(c10::Symbol::fromQualString("onnx::" + op), inputs, 1);
  graph->insertNode(result);
  return result;
}

Value *Lowering::op(const std::string &name, const std::vector<Value *> &inputs) {
  return node(name, inputs)->output();
}

void Lowering::result(Node *source, Value *target) {
  target->setType(source->output(0)->type());
  values[source->output(0)] = target;
  for (size_t i = 1; i < source->outputs().size(); ++i)
    require(source->output(i)->uses().empty(), error_kind,
            std::string("additional outputs are unsupported for ") + source->kind().toQualString());
}

std::shared_ptr<Graph> lower_graph(const std::shared_ptr<Graph> &source, int64_t opset) {
  torch::jit::EliminateDeadCode(source);
  Lowering lowering(opset);
  for (auto input : source->inputs()) {
    auto value = lowering.graph->addInput(input->debugName());
    value->setType(input->type());
    lowering.values[input] = value;
  }
  for (auto node : source->nodes()) {
    if (node->kind() == c10::prim::Constant || node->kind() == c10::prim::ListConstruct)
      continue;
    require(node->blocks().empty(), error_kind, "ONNX tracing does not export control-flow blocks");
    if (!lowering.lower_tensor(node) && !lowering.lower_nn(node))
      throw Error(error_kind, "unsupported ONNX operator: " +
                  std::string(node->kind().toQualString()) + " (opset " +
                  std::to_string(opset) + ", %" + node->output(0)->debugName() + ")");
  }
  for (auto output : source->outputs()) {
    auto result = lowering.op("Identity", {lowering.input(output)});
    result->setType(output->type());
    lowering.graph->registerOutput(result);
  }
  lowering.graph->lint();
  return lowering.graph;
}
}
