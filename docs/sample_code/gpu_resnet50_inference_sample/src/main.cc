#include <sys/time.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>


#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"

using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;

std::string RealPath(std::string path) {
  char realPathMem[PATH_MAX] = {0};
  char *realPathRet = nullptr;
  realPathRet = realpath(path.data(), realPathMem);
  if (realPathRet == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }

  std::string realPath(realPathMem);
  return realPath;
}


int load_model(Model *model, std::vector<MSTensor> *model_inputs, std::string mindir_path, int device_id) {
  if (RealPath(mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  auto context = std::make_shared<Context>();
  auto gpu_device_info = std::make_shared<mindspore::GPUDeviceInfo>();
  gpu_device_info->SetDeviceID(device_id);
  gpu_device_info->SetPrecisionMode("fp16");
  context->MutableDeviceInfo().push_back(gpu_device_info);
  mindspore::Graph graph;
  Serialization::Load(mindir_path, ModelType::kMindIR, &graph);

  Status ret = model->Build(GraphCell(graph), context);
  if (ret != kSuccess) {
    std::cout << "ERROR: Build failed." << std::endl;
    return 1;
  }

  *model_inputs = model->GetInputs();
  if (model_inputs->empty()) {
    std::cout << "Invalid model, inputs is empty." << std::endl;
    return 1;
  }
  return 0;
}


int main(int argc, char **argv) {
  std::cout << "===========================================================" << std::endl;
  std::cout << "  Usage: ./main MINDIR_FILEPATH INFER_STEP WARMUP_STEP" << std::endl;
  std::cout << "  For example: ./main /home/vgg.mindir 1000 10" << std::endl;
  std::cout << "===========================================================" << std::endl;

  // parse arguments
  std::string mindir_file = argv[1];
  int infer_step = atoi(argv[2]);
  int warmup_step = atoi(argv[3]);
  std::cout << "The input argumemts: " << std::endl;
  std::cout << "  mindir_file: " << mindir_file << std::endl;
  std::cout << "  infer_step: " << infer_step << std::endl;
  std::cout << "  warmup_step: " << warmup_step << std::endl;

  // load model
  std::cout << "Start to load model.." << std::endl;
  Model model;
  std::vector<MSTensor> model_inputs;
  load_model(&model, &model_inputs, mindir_file, 0);
  std::cout << "Load model successuflly" << std::endl;

  struct timeval start = {0};
  struct timeval end = {0};
  double startTimeMs;
  double endTimeMs;
  double total_time = 0.0;

  // build dummy inputs
  std::vector<MSTensor> inputs;
  std::vector<MSTensor> outputs;
  float *dummy_data = new float[1*3*224*224];
  inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                      dummy_data, 1*3*224*224*sizeof(float));

  // warmup
  std::cout << "Start to warmup.." << std::endl;
  for (int i = 0; i < warmup_step; i++) {
    Status ret = model.Predict(inputs, &outputs);
    if (ret != kSuccess) {
      std::cout << "Predict failed." << std::endl;
      return 1;
    }
  }
  std::cout << "Warmup finished" << std::endl;

  // inference
  std::cout << "Start to infer.." << std::endl;
  for (int i = 0; i < infer_step; i++) {
    gettimeofday(&start, nullptr);
    Status ret = model.Predict(inputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
      std::cout << "Predict failed." << std::endl;
      return 1;
    }
    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    std::cout << "step " << i << " cost " << endTimeMs - startTimeMs << "ms"<< std::endl;
    total_time += endTimeMs - startTimeMs;
  }
  std::cout << "infer finished." << std::endl;

  delete[] dummy_data;
  std::cout << "=================Average inference time: " << total_time / infer_step << " ms" << std::endl;
  return 0;
}
