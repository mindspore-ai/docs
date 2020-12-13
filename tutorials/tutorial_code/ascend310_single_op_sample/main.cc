#include <iostream>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"

namespace ms = mindspore::api;
constexpr auto tensor_add_file = "./tensor_add.mindir";
static const std::vector<float> input_data_1 = {1, 2, 3, 4};
static const std::vector<float> input_data_2 = {2, 3, 4, 5};

int main() {
  ms::Context::Instance().SetDeviceTarget(ms::kDeviceTypeAscend310).SetDeviceID(0);
  auto graph = ms::Serialization::LoadModel(tensor_add_file, ms::ModelType::kMindIR);
  ms::Model tensor_add((ms::GraphCell(graph)));
  ms::Status ret = tensor_add.Build({});
  if (ret != ms::SUCCESS) {
    std::cout << "Build model failed." << std::endl;
    return 1;
  }

  // prepare input
  std::vector<ms::Buffer> outputs;
  std::vector<ms::Buffer> inputs;
  inputs.emplace_back(input_data_1.data(), sizeof(float) * input_data_1.size());
  inputs.emplace_back(input_data_2.data(), sizeof(float) * input_data_2.size());

  // infer
  ret = tensor_add.Predict(inputs, &outputs);
  if (ret != ms::SUCCESS) {
    std::cout << "Predict model failed." << std::endl;
    return 1;
  }

  // print
  for (auto &buffer : outputs) {
    const float *p = reinterpret_cast<const float *>(buffer.Data());
    for (size_t i = 0; i < buffer.DataSize() / sizeof(float); ++i) {
      std::cout << p[i] << std::endl;
    }
  }

  return 0;
}