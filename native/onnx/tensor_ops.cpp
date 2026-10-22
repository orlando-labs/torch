#include "lowering.h"
#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace sputnik_torch::onnx_export {
bool Lowering::lower_tensor(Node *source) {
  const std::string name = source->kind().toQualString();
  auto arg = [&](size_t i) { return source->input(i); };
  auto finish = [&](Value *value) { result(source, value); return true; };
  if (name == "aten::zeros" || name == "aten::ones" || name == "aten::full") {
    auto shape = integer_list(arg(0));
    auto options = at::TensorOptions().dtype(scalar_type(source->output(0))).device(at::kCPU);
    if (name == "aten::zeros") return finish(constant(at::zeros(shape, options)));
    if (name == "aten::ones") return finish(constant(at::ones(shape, options)));
    auto value = literal(arg(1));
    auto fill = value.isInt() ? at::Scalar(value.toInt()) : at::Scalar(value.toDouble());
    return finish(constant(at::full(shape, fill, options)));
  }
  if (name == "aten::zeros_like" || name == "aten::ones_like" || name == "aten::full_like") {
    const auto dtype = scalar_type(source->output(0));
    auto value = name == "aten::full_like" ? literal(arg(1)) :
      c10::IValue(name == "aten::ones_like" ? int64_t{1} : int64_t{0});
    auto fill = value.isInt() ? at::scalar_tensor(value.toInt(), at::TensorOptions().dtype(dtype)) :
      at::scalar_tensor(value.toDouble(), at::TensorOptions().dtype(dtype));
    auto node = this->node("ConstantOfShape", {op("Shape", {input(arg(0))})});
    node->t_(c10::attr::value, fill.reshape({1}));
    return finish(node->output());
  }
  if (name == "aten::arange") {
    const auto dtype = scalar_type(source->output(0));
    const auto schema = source->maybeSchema();
    Value *start = scalar(0, dtype), *step = scalar(1, dtype), *end = nullptr;
    for (size_t i = 0; schema && i < schema->arguments().size(); ++i) {
      const auto &key = schema->arguments()[i].name();
      if (key == "start") start = typed(arg(i), dtype);
      if (key == "end") end = typed(arg(i), dtype);
      if (key == "step") step = typed(arg(i), dtype);
    }
    require(end != nullptr, error_kind, "arange requires a constant end");
    return finish(op("Range", {start, end, step}));
  }
  static const std::unordered_map<std::string, std::string> unary = {
    {"aten::relu", "Relu"}, {"aten::sigmoid", "Sigmoid"}, {"aten::tanh", "Tanh"},
    {"aten::abs", "Abs"}, {"aten::neg", "Neg"}, {"aten::exp", "Exp"},
    {"aten::log", "Log"}, {"aten::sqrt", "Sqrt"}, {"aten::reciprocal", "Reciprocal"},
    {"aten::sin", "Sin"}, {"aten::cos", "Cos"}, {"aten::floor", "Floor"},
    {"aten::ceil", "Ceil"}, {"aten::round", "Round"}, {"aten::erf", "Erf"},
    {"aten::logical_not", "Not"}, {"aten::isnan", "IsNaN"},
    {"aten::detach", "Identity"}, {"aten::clone", "Identity"},
    {"aten::contiguous", "Identity"}, {"aten::alias", "Identity"}
  };
  if (auto found = unary.find(name); found != unary.end())
    return finish(op(found->second, {name == "aten::logical_not" ? cast(input(arg(0)), at::kBool) : input(arg(0))}));
  if (name == "aten::square")
    return finish(op("Mul", {input(arg(0)), input(arg(0))}));
  if (name == "aten::rsqrt")
    return finish(op("Reciprocal", {op("Sqrt", {input(arg(0))})}));

  static const std::unordered_map<std::string, std::string> binary = {
    {"aten::add", "Add"}, {"aten::sub", "Sub"}, {"aten::mul", "Mul"},
    {"aten::div", "Div"}, {"aten::pow", "Pow"}, {"aten::eq", "Equal"},
    {"aten::lt", "Less"}, {"aten::le", "LessOrEqual"}, {"aten::gt", "Greater"},
    {"aten::ge", "GreaterOrEqual"}, {"aten::ne", "Equal"},
    {"aten::logical_and", "And"}, {"aten::logical_or", "Or"},
    {"aten::maximum", "Max"}, {"aten::minimum", "Min"}
  };
  if (auto found = binary.find(name); found != binary.end()) {
    const bool comparison = name == "aten::eq" || name == "aten::ne" || name == "aten::lt" ||
      name == "aten::le" || name == "aten::gt" || name == "aten::ge";
    const bool logical = name == "aten::logical_and" || name == "aten::logical_or";
    auto dtype = scalar_type(arg(0));
    if (arg(1)->type()->cast<c10::TensorType>())
      dtype = c10::promoteTypes(dtype, scalar_type(arg(1)));
    else if (comparison && literal(arg(1)).isDouble() && !c10::isFloatingType(dtype))
      dtype = at::kFloat;
    if (!comparison) dtype = scalar_type(source->output(0));
    if (logical) dtype = at::kBool;
    if (name == "aten::div") {
      require(source->inputs().size() < 3 || absent(arg(2)), error_kind,
              "ONNX div supports true division only");
      dtype = scalar_type(source->output(0));
    }
    auto left = typed(arg(0), dtype);
    auto right = typed(arg(1), dtype);
    if ((name == "aten::add" || name == "aten::sub") && source->inputs().size() == 3) {
      auto alpha = number_value(arg(2));
      if (alpha != 1)
        right = op("Mul", {right, scalar(alpha, dtype)});
    }
    auto op_name = found->second;
    if (dtype == at::kBool && !comparison) {
      if (name == "aten::mul" || name == "aten::minimum") op_name = "And";
      if (name == "aten::add" || name == "aten::maximum") op_name = "Or";
    }
    auto value = op(op_name, {left, right});
    return finish(name == "aten::ne" ? op("Not", {value}) : value);
  }
  if (name == "aten::matmul" || name == "aten::mm" || name == "aten::bmm")
    return finish(op("MatMul", {input(arg(0)), input(arg(1))}));
  if (name == "aten::addmm") {
    auto node = this->node("Gemm", {input(arg(1)), input(arg(2)), input(arg(0))});
    node->f_(c10::Symbol::attr("beta"), number_value(arg(3)));
    node->f_(c10::Symbol::attr("alpha"), number_value(arg(4)));
    return finish(node->output());
  }
  if (name == "aten::to" || name == "aten::_to_copy" || name == "aten::type_as")
    return finish(cast(input(arg(0)), scalar_type(source->output(0))));
  if (name == "aten::reshape" || name == "aten::view") {
    auto node = this->node("Reshape", {input(arg(0)), ints(integer_list(arg(1)))});
    node->i_(c10::Symbol::attr("allowzero"), 1);
    return finish(node->output());
  }
  if (name == "aten::clamp")
    return finish(op("Clip", {input(arg(0)), typed(arg(1), scalar_type(arg(0))),
                              typed(arg(2), scalar_type(arg(0)))}));
  if (name == "aten::flatten") {
    auto start = integer_value(arg(1));
    auto end = integer_value(arg(2));
    const auto rank = tensor_rank(arg(0));
    if (start < 0) start += rank;
    if (end < 0) end += rank;
    if (rank == 0) return finish(op("Reshape", {input(arg(0)), ints({1})}));
    require(start >= 0 && start <= end && end < rank, error_kind, "invalid flatten dimensions");
    auto shape = op("Shape", {input(arg(0))});
    auto section = [&](int64_t first, int64_t last) {
      return op("Slice", {shape, ints({first}), ints({last}), ints({0})});
    };
    auto product = node("ReduceProd", {section(start, end + 1), ints({0})});
    product->i_(c10::Symbol::attr("keepdims"), 1);
    std::vector<Value *> parts;
    if (start > 0) parts.push_back(section(0, start));
    parts.push_back(product->output());
    if (end < rank - 1) parts.push_back(section(end + 1, rank));
    auto target = node("Concat", parts);
    target->i_(c10::Symbol::attr("axis"), 0);
    return finish(op("Reshape", {input(arg(0)), target->output()}));
  }
  if (name == "aten::transpose" || name == "aten::permute" || name == "aten::t") {
    auto rank = tensor_rank(arg(0));
    std::vector<int64_t> permutation(static_cast<size_t>(rank));
    std::iota(permutation.begin(), permutation.end(), 0);
    if (name == "aten::permute") {
      permutation = integer_list(arg(1));
    } else if (name == "aten::t") {
      require(rank <= 2, error_kind, "t requires rank at most 2");
      std::reverse(permutation.begin(), permutation.end());
    } else {
      auto first = integer_value(arg(1));
      auto second = integer_value(arg(2));
      if (first < 0) first += rank;
      if (second < 0) second += rank;
      std::swap(permutation.at(first), permutation.at(second));
    }
    for (auto &axis : permutation) if (axis < 0) axis += rank;
    auto node = this->node("Transpose", {input(arg(0))});
    node->is_(c10::Symbol::attr("perm"), permutation);
    return finish(node->output());
  }
  if (name == "aten::unsqueeze" || name == "aten::squeeze") {
    std::vector<Value *> inputs{input(arg(0))};
    if (source->inputs().size() > 1 && !absent(arg(1))) {
      auto axis = literal(arg(1));
      inputs.push_back(ints(axis.isInt() ? std::vector<int64_t>{axis.toInt()} : axis.toIntVector()));
    }
    return finish(op(name == "aten::unsqueeze" ? "Unsqueeze" : "Squeeze", inputs));
  }
  if (name == "aten::select" || name == "aten::index_select" || name == "aten::embedding") {
    auto axis = name == "aten::embedding" ? 0 : integer_value(arg(1));
    auto indices = name == "aten::embedding" ? input(arg(1)) :
      name == "aten::index_select" ? input(arg(2)) :
      constant(at::scalar_tensor(integer_value(arg(2)), at::TensorOptions().dtype(at::kLong)));
    auto node = this->node("Gather", {input(arg(0)), indices});
    node->i_(c10::Symbol::attr("axis"), axis);
    return finish(node->output());
  }
  if (name == "aten::slice" || name == "aten::narrow") {
    const auto axis = integer_value(arg(1));
    auto start = absent(arg(2)) ? 0 : integer_value(arg(2));
    auto end = absent(arg(3)) ? std::numeric_limits<int64_t>::max() : integer_value(arg(3));
    auto step = name == "aten::narrow" ? 1 : integer_value(arg(4));
    if (name == "aten::narrow") {
      end += start;
      if (start < 0 && end == 0) end = std::numeric_limits<int64_t>::max();
    }
    return finish(op("Slice", {input(arg(0)), ints({start}), ints({end}), ints({axis}), ints({step})}));
  }
  if (name == "aten::repeat")
    return finish(op("Tile", {input(arg(0)), ints(integer_list(arg(1)))}));
  if (name == "aten::expand" || name == "aten::expand_as") {
    Value *shape;
    if (name == "aten::expand_as") {
      shape = op("Shape", {input(arg(1))});
    } else {
      auto target = integer_list(arg(1));
      shape = ints(target);
      if (std::find(target.begin(), target.end(), -1) != target.end()) {
        auto original = op("Shape", {input(arg(0))});
        const auto prefix = static_cast<int64_t>(target.size()) - tensor_rank(arg(0));
        require(prefix >= 0, error_kind, "expand cannot reduce rank");
        if (prefix > 0) {
          auto combined = node("Concat", {ints(std::vector<int64_t>(prefix, 1)), original});
          combined->i_(c10::Symbol::attr("axis"), 0);
          original = combined->output();
        }
        shape = op("Where", {op("Equal", {shape, ints(std::vector<int64_t>(target.size(), -1))}),
                              original, shape});
      }
    }
    return finish(op("Expand", {input(arg(0)), shape}));
  }
  if (name == "aten::cat" || name == "aten::stack") {
    require(arg(0)->node()->kind() == c10::prim::ListConstruct, error_kind,
            "cat/stack requires a traced tensor list");
    std::vector<Value *> inputs;
    const auto axis = integer_value(arg(1));
    for (auto value : arg(0)->node()->inputs())
      inputs.push_back(name == "aten::stack" ? op("Unsqueeze", {input(value), ints({axis})}) : input(value));
    auto node = this->node("Concat", inputs);
    node->i_(c10::Symbol::attr("axis"), axis);
    return finish(node->output());
  }
  if (name == "aten::where") {
    auto dtype = scalar_type(source->output(0));
    return finish(op("Where", {input(arg(0)), typed(arg(1), dtype), typed(arg(2), dtype)}));
  }
  if (name == "aten::masked_fill")
    return finish(op("Where", {input(arg(1)), typed(arg(2), scalar_type(arg(0))), input(arg(0))}));
  if (name == "aten::sum" || name == "aten::mean" || name == "aten::amax" || name == "aten::amin" ||
      name == "aten::any" || name == "aten::all") {
    std::vector<Value *> inputs{cast(input(arg(0)), scalar_type(source->output(0)))};
    bool keepdim = false;
    const auto schema = source->maybeSchema();
    for (size_t i = 1; schema && i < schema->arguments().size(); ++i) {
      const auto &key = schema->arguments()[i].name();
      if (key == "dim" && !absent(arg(i))) {
        auto axes = literal(arg(i));
        inputs.push_back(ints(axes.isInt() ? std::vector<int64_t>{axes.toInt()} : axes.toIntVector()));
      }
      if (key == "keepdim") keepdim = boolean_value(arg(i));
    }
    const std::string op_name = name == "aten::sum" ? "ReduceSum" : name == "aten::mean" ? "ReduceMean" :
      name == "aten::amax" ? "ReduceMax" : name == "aten::amin" ? "ReduceMin" :
      name == "aten::any" ? "ReduceMax" : "ReduceMin";
    if (name == "aten::any" || name == "aten::all")
      inputs[0] = cast(inputs[0], at::kLong);
    auto node = this->node(op_name, inputs);
    node->i_(c10::Symbol::attr("keepdims"), keepdim);
    auto value = node->output();
    if (name == "aten::any" || name == "aten::all") value = cast(value, at::kBool);
    return finish(value);
  }
  if (name == "aten::softmax" || name == "aten::log_softmax") {
    auto node = this->node(name == "aten::softmax" ? "Softmax" : "LogSoftmax",
                          {cast(input(arg(0)), scalar_type(source->output(0)))});
    node->i_(c10::Symbol::attr("axis"), integer_value(arg(1)));
    return finish(node->output());
  }
  if (name == "aten::argmax") {
    auto value = input(arg(0));
    auto axis = absent(arg(1)) ? 0 : integer_value(arg(1));
    if (absent(arg(1))) value = op("Reshape", {value, ints({-1})});
    auto node = this->node("ArgMax", {value});
    node->i_(c10::Symbol::attr("axis"), axis);
    node->i_(c10::Symbol::attr("keepdims"), boolean_value(arg(2)));
    node->i_(c10::Symbol::attr("select_last_index"), 0);
    return finish(node->output());
  }
  return false;
}
}
