# Experience C++ Minimalist Concurrent Reasoning Demo

`Linux` `Windows` `x86` `C++` `Whole Process` `Inference Application` `Data Preparation` `Beginner`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/quick_start/quick_start_server_inference_cpp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial provides a MindSpore Lite parallel inference demo. It demonstrates the basic on-device inference process using C++ by inputting random data, executing inference, and printing the inference result. You can quickly understand how to use inference-related APIs on MindSpore Lite. In this tutorial, the randomly generated data is used as the input data to perform the inference on the MobileNetV2 model and print the output data. The code is stored in the [mindspore/lite/examples/quick_start_server_inference_cpp](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_server_inference_cpp) directory.

The MindSpore Lite parallel inference steps are as follows:

1. Read the model: Read the `.ms` model file converted by the [model conversion tool](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html) from the file system.
2. Create configuration options: Create and configure to save some basic configuration parameters required to build and execute the model.
3. Init a ModelParallelRunner: Initialization: Before executing concurrent inference, you need to call the init interface of ModelParallelRunner to initialize concurrent inference, mainly for model reading, concurrent creation, subgraph segmentation, and operator selection and scheduling. This part will take a lot of time, so it is recommended to initialize it once and perform concurrent inference multiple times.
4. Input data: Before the model is executed, data needs to be filled in the `Input Tensor`.
5. parallel inference: Use Predict of ModelParallelRunner to perform model inference.
6. Obtain the output: After the model execution is complete, you can obtain the inference result by `Output Tensor`.
7. Release the memory: If the MindSpore Lite inference framework is not required, release the created ModelParallelRunner.

![img](../images/server_inference.png)

## Building and Running

### Linux X86

- Environment requirements

    - System environment: Linux x86_64 (Ubuntu 18.04.02LTS is recommended.)
    - Build dependency:
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- Build

  Run the [build script](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_server_inference_cpp/build.sh) in the `mindspore/lite/examples/quick_start_server_inference_cpp` directory to automatically download the MindSpore Lite inference framework library and model files and build the demo.

  ```bash
  bash build.sh
  ```

  > If the MindSpore Lite inference framework fails to be downloaded by using this build script, manually download the MindSpore Lite model inference framework [mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) whose hardware platform is CPU and operating system is Ubuntu-x64, and copy the `libmindspore-lite.so` file in the decompressed lib directory to the `mindspore/lite/examples/quick_start_server_inference_cpp/lib` directory. Also copy the files from `runtime/include` to the `mindspore/lite/examples/quick_start_server_inference_cpp/include` directory.
  >
  > If the MobileNetV2 model fails to be downloaded, manually download the model file [mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms) and copy it to the `mindspore/lite/examples/quick_start_server_inference_cpp/model` directory.
  >
  > After manually downloading and placing the file in the specified location, you need to execute the build.sh script again to complete the compilation.

- Inference

  After the build, go to the `mindspore/lite/examples/quick_start_server_inference_cpp/build` directory and run the following command to experience MindSpore Lite inference on the MobileNetV2 model:

  ```bash
  ./mindspore_quick_start_cpp ../model/mobilenetv2.ms
  ```

  After the execution, the following information is displayed, including the tensor name, tensor size, number of output tensors, and the first 50 pieces of data.

  ```text
  tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
  output data is:1.74225e-05 1.15919e-05 2.02728e-05 0.000106485 0.000124295 0.00140576 0.000185107 0.000762011 1.50996e-05 5.91942e-06 6.61469e-06 3.72883e-06 4.30761e-06 2.38897e-06 1.5163e-05 0.000192663 1.03767e-05 1.31953e-05 6.69638e-06 3.17411e-05 4.00895e-06 9.9641e-06 3.85127e-06 6.25101e-06 9.08853e-06 1.25043e-05 1.71761e-05 4.92751e-06 2.87637e-05 7.46446e-06 1.39375e-05 2.18824e-05 1.08861e-05 2.5007e-06 3.49876e-05 0.000384547 5.70778e-06 1.28909e-05 1.11038e-05 3.53906e-06 5.478e-06 9.76608e-06 5.32172e-06 1.10386e-05 5.35474e-06 1.35796e-05 7.12652e-06 3.10017e-05 4.34154e-06 7.89482e-05 1.79441e-05
  ```

## Init

```c++
// Create and init context, add CPU device info
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
  std::cerr << "New context failed." << std::endl;
  return -1;
}
auto &device_list = context->MutableDeviceInfo();
auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
if (device_info == nullptr) {
  std::cerr << "New CPUDeviceInfo failed." << std::endl;
  return -1;
}
device_list.push_back(device_info);

// Create model
auto model_runner = new (std::nothrow) mindspore::ModelParallelRunner();
if (model_runner == nullptr) {
  std::cerr << "New Model failed." << std::endl;
  return -1;
}
auto runner_config = std::make_shared<mindspore::RunnerConfig>();
runner_config->context = context;
runner_config->workers_num = 2;
// Build model
auto build_ret = model_runner->Init(model_path, runner_config);
if (build_ret != mindspore::kSuccess) {
  delete model_runner;
  std::cerr << "Build model error " << build_ret << std::endl;
  return -1;
}
```

## Parallel predict

ModelParallelRunner predict includes input data injection, inference execution, and output obtaining. In this example, the input data is randomly generated, and the output result is printed after inference.

```c++
// Get Input
auto model_input = model_runner->GetInputs();
// Generate random data as input data.
auto inputs = GenerateInputDataWithRandom(model_input);
// Get Output
std::vector<mindspore::MSTensor> outputs;

// Model Predict
auto predict_ret = model_runner->Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  delete model_runner;
  std::cerr << "Predict error " << predict_ret << std::endl;
  return -1;
}
```

## Memory Release

If the inference process of MindSpore Lite is complete, release the created `ModelParallelRunner`.

```c++
delete model_runner;
```
