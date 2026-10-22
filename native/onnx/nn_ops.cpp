#include "lowering.h"
#include <algorithm>
#include <cmath>

namespace sputnik_torch::onnx_export {
bool Lowering::lower_nn(Node *source) {
  const std::string name = source->kind().toQualString();
  auto arg = [&](size_t i) { return source->input(i); };
  auto finish = [&](Value *value) { result(source, value); return true; };
  if (source->inputs().empty() || !arg(0)->type()->cast<c10::TensorType>()) return false;
  auto x = input(arg(0));
  const auto dtype = scalar_type(arg(0));
  auto scalar = [&](double number) { return this->scalar(number, dtype); };
  auto spatial = [&](Value *value, size_t rank) {
    auto dimensions = integer_list(value);
    if (dimensions.size() == 1) dimensions.resize(rank, dimensions[0]);
    require(dimensions.size() == rank, error_kind, "invalid spatial argument");
    return dimensions;
  };
  if (name == "aten::_convolution") {
    const auto rank = static_cast<size_t>(tensor_rank(arg(1)) - 2);
    const bool transpose = boolean_value(arg(6));
    std::vector<Value *> inputs{x, input(arg(1))};
    if (!absent(arg(2))) inputs.push_back(input(arg(2)));
    auto node = this->node(transpose ? "ConvTranspose" : "Conv", inputs);
    auto pads = spatial(arg(4), rank);
    auto ends = pads;
    pads.insert(pads.end(), ends.begin(), ends.end());
    node->is_(c10::Symbol::attr("pads"), pads);
    node->is_(c10::Symbol::attr("strides"), spatial(arg(3), rank));
    node->is_(c10::Symbol::attr("dilations"), spatial(arg(5), rank));
    node->i_(c10::Symbol::attr("group"), integer_value(arg(8)));
    if (transpose) node->is_(c10::Symbol::attr("output_padding"), spatial(arg(7), rank));
    return finish(node->output());
  }
  if (name == "aten::linear") {
    auto transpose = node("Transpose", {input(arg(1))});
    transpose->is_(c10::Symbol::attr("perm"), {1, 0});
    auto result = op("MatMul", {x, transpose->output()});
    if (!absent(arg(2))) result = op("Add", {result, input(arg(2))});
    return finish(result);
  }
  if (name == "aten::batch_norm") {
    require(!boolean_value(arg(5)), error_kind, "training BatchNorm is unsupported in ONNX export");
    require(!absent(arg(3)) && !absent(arg(4)), error_kind,
            "ONNX BatchNorm requires running mean and variance");
    const auto channels = tensor_shape(arg(3))[0];
    auto scale = absent(arg(1)) ? constant(at::ones({channels}, at::TensorOptions().dtype(dtype))) : input(arg(1));
    auto bias = absent(arg(2)) ? constant(at::zeros({channels}, at::TensorOptions().dtype(dtype))) : input(arg(2));
    auto node = this->node("BatchNormalization", {x, scale, bias, input(arg(3)), input(arg(4))});
    node->f_(c10::Symbol::attr("epsilon"), number_value(arg(7)));
    node->f_(c10::Symbol::attr("momentum"), 1.0 - number_value(arg(6)));
    node->i_(c10::Symbol::attr("training_mode"), 0);
    return finish(node->output());
  }
  if (name == "aten::adaptive_avg_pool1d" || name == "aten::adaptive_avg_pool2d" ||
      name == "aten::adaptive_avg_pool3d" || name == "aten::adaptive_max_pool1d" ||
      name == "aten::adaptive_max_pool2d" || name == "aten::adaptive_max_pool3d") {
    const auto target = integer_list(arg(1));
    const bool average = name.find("avg") != std::string::npos;
    if (std::all_of(target.begin(), target.end(), [](auto size) { return size == 1; }))
      return finish(op(average ? "GlobalAveragePool" : "GlobalMaxPool", {x}));
    auto shape = tensor_shape(arg(0));
    std::vector<int64_t> strides;
    for (size_t i = 0; i < target.size(); ++i) {
      require(shape[i + 2] % target[i] == 0, error_kind,
              "non-global adaptive pooling requires evenly divisible static spatial dimensions");
      strides.push_back(shape[i + 2] / target[i]);
    }
    auto node = this->node(average ? "AveragePool" : "MaxPool", {x});
    node->is_(c10::Symbol::attr("kernel_shape"), strides);
    node->is_(c10::Symbol::attr("strides"), strides);
    return finish(node->output());
  }
  if (name == "aten::max_pool1d" || name == "aten::max_pool2d" || name == "aten::max_pool3d" ||
      name == "aten::max_pool2d_with_indices" || name == "aten::max_pool3d_with_indices" || name == "aten::avg_pool1d" ||
      name == "aten::avg_pool2d" || name == "aten::avg_pool3d") {
    const bool average = name.find("avg") != std::string::npos;
    const auto rank = static_cast<size_t>(tensor_rank(arg(0)) - 2);
    auto kernel = spatial(arg(1), rank);
    auto strides = integer_list(arg(2));
    if (strides.empty()) strides = kernel;
    auto pads = spatial(arg(3), rank);
    auto ends = pads;
    pads.insert(pads.end(), ends.begin(), ends.end());
    auto node = this->node(average ? "AveragePool" : "MaxPool", {x});
    node->is_(c10::Symbol::attr("kernel_shape"), kernel);
    node->is_(c10::Symbol::attr("strides"), strides);
    node->is_(c10::Symbol::attr("pads"), pads);
    node->i_(c10::Symbol::attr("ceil_mode"), boolean_value(arg(average ? 4 : 5)));
    if (average) {
      require(source->inputs().size() < 7 || absent(arg(6)), error_kind,
              "AveragePool divisor_override is unsupported");
      node->i_(c10::Symbol::attr("count_include_pad"), boolean_value(arg(5)));
    } else {
      node->is_(c10::Symbol::attr("dilations"), spatial(arg(4), rank));
    }
    return finish(node->output());
  }
  if (name == "aten::layer_norm" || name == "aten::native_layer_norm") {
    auto shape = integer_list(arg(1));
    auto scale = absent(arg(2)) ? constant(at::ones(shape, at::TensorOptions().dtype(dtype))) : input(arg(2));
    auto bias = absent(arg(3)) ? constant(at::zeros(shape, at::TensorOptions().dtype(dtype))) : input(arg(3));
    auto node = this->node("LayerNormalization", {x, scale, bias});
    node->i_(c10::Symbol::attr("axis"), -static_cast<int64_t>(shape.size()));
    node->f_(c10::Symbol::attr("epsilon"), number_value(arg(4)));
    return finish(node->output());
  }
  if (name == "aten::group_norm") {
    const auto groups = integer_value(arg(1));
    auto shape = op("Shape", {x});
    auto reshaped = op("Reshape", {x, ints({0, groups, -1})});
    auto scale = constant(at::ones({groups}, at::TensorOptions().dtype(dtype)));
    auto bias = constant(at::zeros({groups}, at::TensorOptions().dtype(dtype)));
    auto normal = node("InstanceNormalization", {reshaped, scale, bias});
    normal->f_(c10::Symbol::attr("epsilon"), number_value(arg(4)));
    auto value = op("Reshape", {normal->output(), shape});
    std::vector<int64_t> affine_shape(tensor_rank(arg(0)), 1);
    affine_shape[1] = -1;
    if (!absent(arg(2))) value = op("Mul", {value, op("Reshape", {input(arg(2)), ints(affine_shape)})});
    if (!absent(arg(3))) value = op("Add", {value, op("Reshape", {input(arg(3)), ints(affine_shape)})});
    return finish(value);
  }
  if (name == "aten::pad" || name == "aten::constant_pad_nd" ||
      name.starts_with("aten::reflection_pad") || name.starts_with("aten::replication_pad")) {
    auto padding = integer_list(arg(1));
    auto rank = tensor_rank(arg(0));
    std::vector<int64_t> pads(static_cast<size_t>(rank * 2), 0);
    for (size_t i = 0; i < padding.size() / 2; ++i) {
      pads[rank - 1 - i] = padding[i * 2];
      pads[rank * 2 - 1 - i] = padding[i * 2 + 1];
    }
    std::string mode = name == "aten::pad" ? std::string(literal(arg(2)).toStringRef()) :
      name == "aten::constant_pad_nd" ? "constant" :
      name.starts_with("aten::reflection_pad") ? "reflect" : "replicate";
    require(mode == "constant" || mode == "reflect" || mode == "replicate", error_kind,
            "ONNX pad supports constant, reflect and replicate modes");
    auto fill = name == "aten::constant_pad_nd" ? typed(arg(2), dtype) :
      name == "aten::pad" && !absent(arg(3)) ? typed(arg(3), dtype) : scalar(0);
    auto node = this->node("Pad", {x, ints(pads), fill});
    node->s_(c10::Symbol::attr("mode"), mode == "replicate" ? "edge" : mode);
    return finish(node->output());
  }
  if (name == "aten::dropout") {
    require(!boolean_value(arg(2)), error_kind, "training Dropout is unsupported in ONNX export");
    return finish(op("Identity", {x}));
  }
  if (name == "aten::gelu") {
    auto mode = literal(arg(1)).toStringRef();
    Value *activation;
    if (mode == "none") {
      activation = op("Erf", {op("Div", {x, scalar(std::sqrt(2.0))})});
    } else {
      require(mode == "tanh", error_kind, "unsupported GELU approximation");
      auto cube = op("Mul", {op("Mul", {x, x}), x});
      activation = op("Tanh", {op("Mul", {scalar(std::sqrt(2.0 / std::acos(-1.0))),
                       op("Add", {x, op("Mul", {scalar(0.044715), cube})})})});
    }
    return finish(op("Mul", {op("Mul", {x, scalar(0.5)}), op("Add", {scalar(1), activation})}));
  }
  if (name == "aten::silu") return finish(op("Mul", {x, op("Sigmoid", {x})}));
  if (name == "aten::prelu") {
    std::vector<int64_t> shape(tensor_rank(arg(0)), 1);
    if (shape.size() > 1) shape[1] = -1;
    return finish(op("PRelu", {x, op("Reshape", {input(arg(1)), ints(shape)})}));
  }
  if (name == "aten::threshold")
    return finish(op("Where", {op("Greater", {x, typed(arg(1), dtype)}), x, typed(arg(2), dtype)}));
  if (name == "aten::mish") return finish(op("Mul", {x, op("Tanh", {op("Softplus", {x})})}));
  if (name == "aten::log_sigmoid") return finish(op("Neg", {op("Softplus", {op("Neg", {x})})}));
  if (name == "aten::softplus") {
    auto beta = scalar(number_value(arg(1)));
    auto scaled = op("Mul", {x, beta});
    auto value = op("Div", {op("Softplus", {scaled}), beta});
    return finish(op("Where", {op("Greater", {scaled, scalar(number_value(arg(2)))}), x, value}));
  }
  if (name == "aten::leaky_relu" || name == "aten::elu" || name == "aten::selu") {
    auto node = this->node(name == "aten::leaky_relu" ? "LeakyRelu" : name == "aten::elu" ? "Elu" : "Selu", {x});
    if (name != "aten::selu") node->f_(c10::Symbol::attr("alpha"), number_value(arg(1)));
    if (name == "aten::elu")
      require(number_value(arg(2)) == 1 && number_value(arg(3)) == 1, error_kind,
              "ELU export requires unit scale and input_scale");
    return finish(node->output());
  }
  if (name == "aten::hardtanh")
    return finish(op("Clip", {x, typed(arg(1), dtype), typed(arg(2), dtype)}));
  if (name == "aten::hardsigmoid" || name == "aten::hardswish") {
    auto clipped = op("Clip", {op("Add", {x, scalar(3)}), scalar(0), scalar(6)});
    auto value = op("Div", {clipped, scalar(6)});
    return finish(name == "aten::hardswish" ? op("Mul", {x, value}) : value);
  }
  if (name == "aten::softshrink" || name == "aten::hardshrink") {
    auto threshold = typed(arg(1), dtype);
    auto positive = op("Greater", {x, threshold});
    auto negative = op("Less", {x, op("Neg", {threshold})});
    if (name == "aten::hardshrink")
      return finish(op("Where", {op("Or", {positive, negative}), x, scalar(0)}));
    return finish(op("Where", {positive, op("Sub", {x, threshold}),
      op("Where", {negative, op("Add", {x, threshold}), scalar(0)})}));
  }
  return false;
}
}
