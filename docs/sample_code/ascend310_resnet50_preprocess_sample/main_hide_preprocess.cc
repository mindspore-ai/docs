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

namespace ms = mindspore;
constexpr auto resnet_file = "./model/resnet50_imagenet.mindir";
constexpr auto image_path = "./test_data/ILSVRC2012_val_00002138.JPEG";

size_t GetMax(ms::MSTensor data);
ms::MSTensor ReadFile(const std::string &file);

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

  // infer
  std::vector<MSTensor> inputs = {ReadFile(image_path)};
  std::vector<MSTensor> outputs;
  ret = resnet50.PredictWithPreprocess(inputs, &outputs);
  if (ret.IsError()) {
    std::cout << "ERROR: PredictWithPreprocess failed." << std::endl;
    return 1;
  }

  // print infer result
  std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;

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

ms::MSTensor ReadFile(const std::string &file) {
  if (file.empty()) {
    std::cout << "Pointer file is nullptr" << std::endl;
    return ms::MSTensor();
  }

  std::ifstream ifs(file);
  if (!ifs.good()) {
    std::cout << "File: " << file << " is not exist" << std::endl;
    return ms::MSTensor();
  }

  if (!ifs.is_open()) {
    std::cout << "File: " << file << "open failed" << std::endl;
    return ms::MSTensor();
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  ms::MSTensor buffer(file, ms::DataType::kNumberTypeUInt8, {static_cast<int64_t>(size)}, nullptr, size);

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buffer.MutableData()), size);
  ifs.close();

  return buffer;
}
