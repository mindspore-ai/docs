/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"

namespace ms = mindspore;
constexpr auto tensor_add_file = "./tensor_add.mindir";
static const std::vector<float> input_data_1 = {1, 2, 3, 4};
static const std::vector<float> input_data_2 = {2, 3, 4, 5};

int main() {
  // set context
  auto context = std::make_shared<ms::Context>();
  auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
  ascend310_info->SetDeviceID(0);
  context->MutableDeviceInfo().push_back(ascend310_info);

  // define model
  ms::Graph graph;
  ms::Status ret = ms::Serialization::Load(tensor_add_file, ms::ModelType::kMindIR, &graph);
  if (ret != ms::kSuccess) {
    std::cout << "Load model failed." << std::endl;
    return 1;
  }
  ms::Model tensor_add;

  // build model
  ret = tensor_add.Build(ms::GraphCell(graph), context);
  if (ret != ms::kSuccess) {
    std::cout << "Build model failed." << std::endl;
    return 1;
  }

  // get model inputs
  std::vector<ms::MSTensor> origin_inputs = tensor_add.GetInputs();
  if (origin_inputs.size() != 2) {
    std::cout << "Invalid model inputs size " << origin_inputs.size() << std::endl;
    return 1;
  }

  // prepare input
  std::vector<ms::MSTensor> outputs;
  std::vector<ms::MSTensor> inputs;
  inputs.emplace_back(origin_inputs[0].Name(), origin_inputs[0].DataType(), origin_inputs[0].Shape(),
                      input_data_1.data(), sizeof(float) * input_data_1.size());
  inputs.emplace_back(origin_inputs[1].Name(), origin_inputs[1].DataType(), origin_inputs[1].Shape(),
                      input_data_2.data(), sizeof(float) * input_data_2.size());

  // infer
  ret = tensor_add.Predict(inputs, &outputs);
  if (ret != ms::kSuccess) {
    std::cout << "Predict model failed." << std::endl;
    return 1;
  }

  // print
  for (auto &buffer : outputs) {
    const float *p = reinterpret_cast<const float *>(buffer.MutableData());
    for (size_t i = 0; i < buffer.DataSize() / sizeof(float); ++i) {
      std::cout << p[i] << std::endl;
    }
  }

  return 0;
}
