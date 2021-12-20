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

#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/api/types.h"

namespace ms = mindspore;
constexpr auto resnet_file = "./model/resnet50_imagenet_preprocess.mindir";
constexpr auto image_file = "./test_data/ILSVRC2012_val_00002138.JPEG";

size_t GetMax(ms::MSTensor data);

int main() {
  // set context
  auto context = std::make_shared<ms::Context>();
  auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
  ascend310_info->SetDeviceID(0);
  context->MutableDeviceInfo().push_back(ascend310_info);

  // define model
  ms::Graph graph;
  ms::Status ret = ms::Serialization::Load(resnet_file, ms::ModelType::kMindIR, &graph);
  if (ret != ms::kSuccess) {
    std::cout << "Load model failed." << std::endl;
    return 1;
  }

  // build model
  ms::Model resnet50;
  ret = resnet50.Build(ms::GraphCell(graph), context);
  if (ret != ms::kSuccess) {
    std::cout << "Build model failed." << std::endl;
    return 1;
  }

  // get model info
  std::vector<ms::MSTensor> model_inputs = resnet50.GetInputs();
  if (model_inputs.empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
  }
  if (!resnet50.HasPreprocess()) {
    std::cout << "data preprocess not exists in MindIR" << std::endl;
    return 1;
  }

  // infer
  std::vector<std::vector<ms::MSTensor>> inputs;
  ms::MSTensor *t1 = ms::MSTensor::CreateTensorFromFile(image_file);
  inputs = {{*t1}};
  std::vector<ms::MSTensor> outputs;
  ret = resnet50.PredictWithPreprocess(inputs, &outputs);
  if (ret.IsError()) {
    std::cout << "ERROR: PredictWithPreprocess failed." << std::endl;
    return 1;
  }

  // print infer result
  std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;
  ms::MSTensor::DestroyTensorPtr(t1);

  return 0;
}

size_t GetMax(ms::MSTensor data) {
  float max_value = -1;
  size_t max_idx = 0;
  const float *p = reinterpret_cast<const float *>(data.MutableData());
  for (size_t i = 0; i < data.DataSize() / sizeof(float); ++i) {
    if (p[i] > max_value) {
      max_value = p[i];
      max_idx = i;
    }
  }
  return max_idx;
}
