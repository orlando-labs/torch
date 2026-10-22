#include <onnxruntime_cxx_api.h>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <iterator>

namespace {
ONNXTensorElementDataType ort_dtype(at::ScalarType type) {
  switch (type) {
  case at::kFloat: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  case at::kDouble: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  case at::kHalf: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  case at::kBFloat16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  case at::kLong: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case at::kInt: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  case at::kShort: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  case at::kChar: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  case at::kByte: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  case at::kBool: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  default: throw std::runtime_error("unsupported reference dtype");
  }
}
}

int main(int argc, char **argv) {
  try {
    TORCH_CHECK(argc >= 3, "usage: onnx_runtime MODEL REFERENCES [CASE ...]");
    at::NoGradGuard no_grad;
    std::ifstream stream(argv[2], std::ios::binary);
    TORCH_CHECK(stream.good(), "cannot open reference checkpoint");
    std::vector<char> bytes{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    auto references = torch::pickle_load(bytes).toGenericDict();
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "sputnik_torch-export-test");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    Ort::Session session(env, argv[1], options);
    Ort::AllocatorWithDefaultOptions allocator;
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<std::string> input_names, output_names;
    for (size_t i = 0; i < session.GetInputCount(); ++i)
      input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < session.GetOutputCount(); ++i)
      output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    std::vector<const char *> inputs, outputs;
    for (auto &name : input_names) inputs.push_back(name.c_str());
    for (auto &name : output_names) outputs.push_back(name.c_str());
    std::vector<std::string> cases;
    if (argc == 3) cases = {"1", "3"};
    for (int i = 3; i < argc; ++i) cases.emplace_back(argv[i]);
    double largest_error = 0;
    for (const auto &test_case : cases) {
      std::vector<at::Tensor> tensors;
      std::vector<Ort::Value> values;
      for (const auto &name : input_names) {
        auto tensor = references.at(c10::IValue(name + "." + test_case)).toTensor().cpu().contiguous();
        tensors.push_back(tensor);
        auto shape = tensor.sizes().vec();
        values.push_back(Ort::Value::CreateTensor(memory, tensor.data_ptr(), tensor.nbytes(),
            shape.data(), shape.size(), ort_dtype(tensor.scalar_type())));
      }
      auto results = session.Run(Ort::RunOptions{nullptr}, inputs.data(), values.data(), values.size(),
                                  outputs.data(), outputs.size());
      for (size_t i = 0; i < results.size(); ++i) {
        auto expected = references.at(c10::IValue(output_names[i] + "." + test_case)).toTensor().cpu();
        auto info = results[i].GetTensorTypeAndShapeInfo();
        TORCH_CHECK(info.GetShape() == expected.sizes().vec(), "output shape differs: ", output_names[i]);
        TORCH_CHECK(info.GetElementType() == ort_dtype(expected.scalar_type()), "output dtype differs");
        auto actual = at::from_blob(const_cast<void *>(results[i].GetTensorRawData()), info.GetShape(),
                                    at::TensorOptions().dtype(expected.scalar_type()));
        if (expected.is_floating_point()) {
          TORCH_CHECK(at::allclose(actual, expected, 1e-4, 1e-5, false),
                      "ONNX output differs: ", output_names[i], ", case ", test_case);
          auto error = (actual.to(at::kDouble) - expected.to(at::kDouble)).abs().max().item<double>();
          if (std::isfinite(error)) largest_error = std::max(largest_error, error);
        } else {
          TORCH_CHECK(at::equal(actual, expected), "integer/bool ONNX output differs");
        }
      }
    }
    std::cout << "PASS ONNX Runtime " << argv[1] << " | " << cases.size()
              << " cases | max error " << largest_error << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
