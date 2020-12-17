#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "minddata/dataset/include/minddata_eager.h"

namespace ms = mindspore::api;
constexpr auto resnet_file = "./model/resnet50_imagenet.mindir";
constexpr auto image_path = "./test_data";

std::vector<std::string> GetAllFiles(std::string_view dir_name);
DIR *OpenDir(std::string_view dir_name);
std::string RealPath(std::string_view path);
std::shared_ptr<ms::Tensor> ReadFile(const std::string &file);
size_t GetMax(const ms::Buffer &data);

int main() {
  ms::Context::Instance()
      .SetDeviceTarget(ms::kDeviceTypeAscend310)
      .SetDeviceID(0);
  auto graph =
      ms::Serialization::LoadModel(resnet_file, ms::ModelType::kMindIR);
  ms::Model resnet50((ms::GraphCell(graph)));
  ms::Status ret = resnet50.Build({});
  if (ret != ms::SUCCESS) {
    std::cout << "Build model failed." << std::endl;
    return 1;
  }

  ms::MindDataEager compose({mindspore::dataset::vision::Decode(),
                             mindspore::dataset::vision::Resize({256}),
                             mindspore::dataset::vision::Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255},
                                                                   {0.229 * 255, 0.224 * 255, 0.225 * 255}),
                             mindspore::dataset::vision::CenterCrop({224, 224}),
                             mindspore::dataset::vision::HWC2CHW(),
                            });
  std::vector<std::string> images = GetAllFiles(image_path);
  for (const auto &image_file : images) {
    // prepare input
    std::vector<ms::Buffer> outputs;
    std::vector<ms::Buffer> inputs;

    auto origin_image = ReadFile(image_file);
    auto img = compose(origin_image);
    inputs.emplace_back(img->Data(), img->DataSize());

    // infer
    ret = resnet50.Predict(inputs, &outputs);
    if (ret != ms::SUCCESS) {
      std::cout << "Predict model failed." << std::endl;
      return 1;
    }

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
  char *real_path_ret = nullptr;
  real_path_ret = realpath(path.data(), real_path_mem);

  if (real_path_ret == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }

  std::string real_path(real_path_mem);
  return real_path;
}

std::shared_ptr<ms::Tensor> ReadFile(const std::string &file) {
  std::shared_ptr<ms::Tensor> buffer = std::make_shared<ms::Tensor>();
  if (file.empty()) {
    std::cout << "Pointer file is nullptr" << std::endl;
    return buffer;
  }

  std::ifstream ifs(file);
  if (!ifs.good()) {
    std::cout << "File: " << file << " is not exist" << std::endl;
    return buffer;
  }

  if (!ifs.is_open()) {
    std::cout << "File: " << file << "open failed" << std::endl;
    return buffer;
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  buffer->ResizeData(size);
  if (buffer->DataSize() != size) {
    std::cout << "Malloc buf failed, file: " << file << std::endl;
    ifs.close();
    return buffer;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buffer->MutableData()), size);
  ifs.close();

  buffer->SetDataType(ms::DataType::kMsUint8);
  buffer->SetShape({static_cast<int64_t>(size)});
  return buffer;
}

size_t GetMax(const ms::Buffer &data) {
  float max_value = -1;
  size_t max_idx = 0;
  const float *p = reinterpret_cast<const float *>(data.Data());
  for (size_t i = 0; i < data.DataSize() / sizeof(float); ++i) {
    if (p[i] > max_value) {
      max_value = p[i];
      max_idx = i;
    }
    return max_idx;
  }
}
