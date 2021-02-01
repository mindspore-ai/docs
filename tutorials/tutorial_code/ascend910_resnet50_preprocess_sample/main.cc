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
#include "include/minddata/dataset/include/execute.h"
#include "include/minddata/dataset/include/vision.h"

namespace ms = mindspore;
constexpr auto resnet_file = "./model/resnet50_imagenet.mindir";
constexpr auto image_path = "./test_data";

std::vector<std::string> GetAllFiles(std::string_view dir_name);
DIR *OpenDir(std::string_view dir_name);
std::string RealPath(std::string_view path);
ms::MSTensor ReadFile(const std::string &file);
size_t GetMax(ms::MSTensor data);

int main() {
  // set context
  ms::GlobalContext::SetGlobalDeviceTarget(ms::kDeviceTypeAscend910);
  ms::GlobalContext::SetGlobalDeviceID(0);
  // define model
  auto graph = ms::Serialization::LoadModel(resnet_file, ms::ModelType::kMindIR);
  ms::Model resnet50((ms::GraphCell(graph)));
  // build model
  ms::Status ret = resnet50.Build();
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
  // define preprocessor
  ms::dataset::Execute preprocessor({ms::dataset::vision::Decode(),
                                     ms::dataset::vision::Resize({256}),
                                     ms::dataset::vision::Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255},
                                                                    {0.229 * 255, 0.224 * 255, 0.225 * 255}),
                                     ms::dataset::vision::CenterCrop({224, 224}),
                                     ms::dataset::vision::HWC2CHW(),
                                    });
  std::vector<std::string> images = GetAllFiles(image_path);
  for (const auto &image_file : images) {
    // prepare input
    std::vector<ms::MSTensor> outputs;
    std::vector<ms::MSTensor> inputs;
    // read image file and preprocess
    auto image = ReadFile(image_file);
    ret = preprocessor(image, &image);
    if (ret != ms::kSuccess) {
      std::cout << "Image preprocess failed." << std::endl;
      return 1;
    }

    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                        image.Data().get(), image.DataSize());
    // infer
    ret = resnet50.Predict(inputs, &outputs);
    if (ret != ms::kSuccess) {
      std::cout << "Predict model failed." << std::endl;
      return 1;
    }
    // print infer result
    std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;
  }

  return 0;
}

std::vector<std::string> GetAllFiles(std::string_view dir_name) {
  struct dirent *filename;
  DIR *dir = OpenDir(dir_name);
  if (dir == nullptr) {
    return {};
  }

  /* read all the files in the dir ~ */
  std::vector<std::string> res;
  while ((filename = readdir(dir)) != nullptr) {
    std::string d_name = std::string(filename->d_name);
    // get rid of "." and ".."
    if (d_name == "." || d_name == ".." || filename->d_type != DT_REG)
      continue;
    res.emplace_back(std::string(dir_name) + "/" + filename->d_name);
  }

  std::sort(res.begin(), res.end());
  return res;
}

DIR *OpenDir(std::string_view dir_name) {
  // check the parameter !
  if (dir_name.empty()) {
    std::cout << " dir_name is null ! " << std::endl;
    return nullptr;
  }

  std::string real_path = RealPath(dir_name);

  // check if dir_name is a valid dir
  struct stat s;
  lstat(real_path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    std::cout << "dir_name is not a valid directory !" << std::endl;
    return nullptr;
  }

  DIR *dir;
  dir = opendir(real_path.c_str());
  if (dir == nullptr) {
    std::cout << "Can not open dir " << dir_name << std::endl;
    return nullptr;
  }
  return dir;
}

std::string RealPath(std::string_view path) {
  char real_path_mem[PATH_MAX] = {0};
  char *real_path_ret = realpath(path.data(), real_path_mem);

  if (real_path_ret == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }

  return std::string(real_path_mem);
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
