# Experiencing C++ Simplified Inference Demo

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/infer/quick_start_cpp.md)

> MindSpore has unified the inference API. If you want to continue to use the MindSpore Lite independent API for inference, you can refer to the [document](https://www.mindspore.cn/lite/docs/en/r1.3/quick_start/quick_start_cpp.html).

## Overview

This tutorial provides a MindSpore Lite inference demo. It demonstrates the basic on-device inference process using C++ by inputting random data, executing inference, and printing the inference result. You can quickly understand how to use inference-related APIs on MindSpore Lite. In this tutorial, the randomly generated data is used as the input data to perform the inference on the MobileNetV2 model and print the output data. The code is stored in the [mindspore/lite/examples/quick_start_cpp](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/quick_start_cpp) directory.

The MindSpore Lite inference steps are as follows:

1. Read the model: Read the `.ms` model file converted by the [model conversion tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html) from the file system.
2. Create and configure context: Create and configure [Context](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Context.html#class-context) to save some basic configuration parameters required to build and execute the model.
3. Create, load and build a model: Use [Build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build) of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html#class-model) to create and build the model, and configure the [Context](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Context.html#class-context) obtained in the previous step. In the model loading phase, the file cache is parsed into a runtime model. In the model building phase, subgraph partition, operator selection and scheduling are performed, which will take a long time. Therefore, it is recommended that the model should be created once, built once, and performed for multiple times.
4. Input data: Before the model is executed, data needs to be filled in the `Input Tensor`.
5. Perform inference: Use [Predict](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#predict) of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html#class-model) to perform model inference.
6. Obtain the output: After the model execution is complete, you can obtain the inference result by `Output Tensor`.
7. Release the memory: If the MindSpore Lite inference framework is not required, release the created [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html#class-model).

![img](../images/lite_runtime.png)

> To view the advanced usage of MindSpore Lite, see [Using Runtime to Perform Inference (C++)](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/runtime_cpp.html).

## Building and Running

### Linux x86

- Environment requirements

    - System environment: Linux x86_64 (Ubuntu 18.04.02LTS is recommended.)
    - Build dependency:
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- Build

  Run the [build script](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/quick_start_cpp/build.sh) in the `mindspore/lite/examples/quick_start_cpp` directory to automatically download the MindSpore Lite inference framework library and model files and build the demo.

  ```bash
  bash build.sh
  ```

  > If the MindSpore Lite inference framework fails to be downloaded by using this build script, manually download the MindSpore Lite model inference framework [mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) whose hardware platform is CPU and operating system is Ubuntu-x64, and copy the `libmindspore-lite.so` file in the decompressed lib directory to the `mindspore/lite/examples/quick_start_cpp/lib` directory. Also copy the files from `runtime/include` to the `mindspore/lite/examples/quick_start_cpp/include` directory, and copy the `libmindspore_glog.so.0` file from the `runtime/third_party/glog` directory to the `libmindspore_glog.so` file in `mindspore/ lite/examples/quick_start_cpp/lib` directory.
  >
  > If the MobileNetV2 model fails to be downloaded, manually download the model file [mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms) and copy it to the `mindspore/lite/examples/quick_start_cpp/model` directory.
  >
  > After manually downloading and placing the file in the specified location, you need to execute the build.sh script again to complete the compilation.

- Inference

  After the build, go to the `mindspore/lite/examples/quick_start_cpp/build` directory and run the following command to experience MindSpore Lite inference on the MobileNetV2 model:

  ```bash
  ./mindspore_quick_start_cpp ../model/mobilenetv2.ms
  ```

  After the execution, the following information is displayed, including the tensor name, tensor size, number of output tensors, and the first 50 pieces of data.

  ```text
  tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
  output data is:1.74225e-05 1.15919e-05 2.02728e-05 0.000106485 0.000124295 0.00140576 0.000185107 0.000762011 1.50996e-05 5.91942e-06 6.61469e-06 3.72883e-06 4.30761e-06 2.38897e-06 1.5163e-05 0.000192663 1.03767e-05 1.31953e-05 6.69638e-06 3.17411e-05 4.00895e-06 9.9641e-06 3.85127e-06 6.25101e-06 9.08853e-06 1.25043e-05 1.71761e-05 4.92751e-06 2.87637e-05 7.46446e-06 1.39375e-05 2.18824e-05 1.08861e-05 2.5007e-06 3.49876e-05 0.000384547 5.70778e-06 1.28909e-05 1.11038e-05 3.53906e-06 5.478e-06 9.76608e-06 5.32172e-06 1.10386e-05 5.35474e-06 1.35796e-05 7.12652e-06 3.10017e-05 4.34154e-06 7.89482e-05 1.79441e-05
  ```

### Windows

- Environment requirements

    - System environment: 64-bit Windows 7 or 64-bit Windows 10
    - Build dependency:
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [MinGW GCC](https://sourceforge.net/projects/mingw-w64/files/ToolchainstargettingWin64/PersonalBuilds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z/download) = 7.3.0

- Build

    - Download the library: Manually download the MindSpore Lite model inference framework [mindspore-lite-{version}-win-x64.zip](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) whose hardware platform is CPU and operating system is Windows-x64. Copy all the files in the decompressed `runtime/lib` directory to the `mindspore/lite/examples/quick_start_cpp/lib` project directory, and change the include directory to the `mindspore/lite/examples/quick_start_cpp/include` project directory. (Note: The `lib` and `include` directories under the project need to be created manually)

    - Download the model: Manually download the model file [mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms) and copy it to the `mindspore/lite/examples/quick_start_cpp/model` directory.

    - Build the demo: Run the [build script](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/quick_start_cpp/build.bat) in the `mindspore/lite/examples/quick_start_cpp` directory to automatically download related files and build the Demo.

  ```bash
  call build.bat
  ```

- Inference

  After the build, go to the `mindspore/lite/examples/quick_start_cpp/build` directory and run the following command to experience MindSpore Lite inference on the MobileNetV2 model:

  ```bash
  set PATH=..\lib;%PATH%
  call mindspore_quick_start_cpp.exe ..\model\mobilenetv2.ms
  ```

  After the execution, the following information is displayed, including the tensor name, tensor size, number of output tensors, and the first 50 pieces of data.

  ```text
  tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
  output data is:1.74225e-05 1.15919e-05 2.02728e-05 0.000106485 0.000124295 0.00140576 0.000185107 0.000762011 1.50996e-05 5.91942e-06 6.61469e-06 3.72883e-06 4.30761e-06 2.38897e-06 1.5163e-05 0.000192663 1.03767e-05 1.31953e-05 6.69638e-06 3.17411e-05 4.00895e-06 9.9641e-06 3.85127e-06 6.25101e-06 9.08853e-06 1.25043e-05 1.71761e-05 4.92751e-06 2.87637e-05 7.46446e-06 1.39375e-05 2.18824e-05 1.08861e-05 2.5007e-06 3.49876e-05 0.000384547 5.70778e-06 1.28909e-05 1.11038e-05 3.53906e-06 5.478e-06 9.76608e-06 5.32172e-06 1.10386e-05 5.35474e-06 1.35796e-05 7.12652e-06 3.10017e-05 4.34154e-06 7.89482e-05 1.79441e-05
  ```

## CMake Integration

The following is the sample code when integrating `libmindspore-lite.a` static library through CMake.

> When CMake integrates the `libmindspore-lite.a` static library, the `-Wl,--whole-archive` option needs to be passed to the linker.
>
> In addition, the build option for stack protection `-fstack-protector-strong` is added during the build of MindSpore Lite. Therefore, the `ssp` library in MinGW needs to be linked on the Windows platform.
>
> In addition, the support of processing .so file is added during the build of MindSpore Lite. Therefore, the `dl` library needs to be linked on the Linux platform.

```cmake
cmake_minimum_required(VERSION 3.18.3)
project(QuickStartCpp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

# Add the directory to include search path
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add the directory to linker search path
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

file(GLOB_RECURSE QUICK_START_CXX ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_executable(mindspore_quick_start_cpp ${QUICK_START_CXX})

target_link_libraries(
        mindspore_quick_start_cpp
        -Wl,--whole-archive libmindspore-lite.a -Wl,--no-whole-archive
        pthread
)

# Due to the increased compilation options for stack protection,
# it is necessary to target link ssp library when Use the static library in Windows.
if(WIN32)
    target_link_libraries(
            mindspore_quick_start_cpp
            ssp
    )
else()
    target_link_libraries(
            mindspore_quick_start_cpp
            dl
    )
endif()
```

## Model Reading

Read the MindSpore Lite model from the file system and store it in the memory buffer.

```c++
// Read model file.
size_t size = 0;
char *model_buf = ReadFile(model_path, &size);
if (model_buf == nullptr) {
  std::cerr << "Read model file failed." << std::endl;
  return -1;
}
```

## Creating and Configuring Context

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
```

## Model Creating Loading and Building

Use [Build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build) of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html#class-model) to load the model directly from the memory buffer and build the model.  

```c++
// Create model
auto model = new (std::nothrow) mindspore::Model();
if (model == nullptr) {
  std::cerr << "New Model failed." << std::endl;
  return -1;
}
// Build model
auto build_ret = model->Build(model_buf, size, mindspore::kMindIR, context);
delete[](model_buf);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model error " << build_ret << std::endl;
  return -1;
}
```

There is another method that uses [Load](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#load) of [Serialization](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Serialization.html#class-serialization) to load [Graph](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Graph.html#class-graph) and use [Build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build) of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html#class-model) to build the model.

```c++
// Load graph.
mindspore::Graph graph;
auto load_ret = mindspore::Serialization::Load(model_buf, size, mindspore::kMindIR, &graph);
delete[](model_buf);
if (load_ret != mindspore::kSuccess) {
  std::cerr << "Load graph file failed." << std::endl;
  return -1;
}

// Create model
auto model = new (std::nothrow) mindspore::Model();
if (model == nullptr) {
  std::cerr << "New Model failed." << std::endl;
  return -1;
}
// Build model
mindspore::GraphCell graph_cell(graph);
auto build_ret = model->Build(graph_cell, context);
if (build_ret != mindspore::kSuccess) {
  delete model;
  std::cerr << "Build model error " << build_ret << std::endl;
  return -1;
}
```

## Model Inference

Model inference includes input data injection, inference execution, and output obtaining. In this example, the input data is randomly generated, and the output result is printed after inference.

```c++
auto inputs = model->GetInputs();
// Generate random data as input data.
auto ret = GenerateInputDataWithRandom(inputs);
if (ret != mindspore::kSuccess) {
  delete model;
  std::cerr << "Generate Random Input Data failed." << std::endl;
  return -1;
}
// Get Output
auto outputs = model->GetOutputs();

// Model Predict
auto predict_ret = model->Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  delete model;
  std::cerr << "Predict model error " << predict_ret << std::endl;
  return -1;
}

// Print Output Tensor Data.
for (auto tensor : outputs) {
  std::cout << "tensor name is:" << tensor.Name() << " tensor size is:" << tensor.DataSize()
            << " tensor elements num is:" << tensor.ElementNum() << std::endl;
  auto out_data = reinterpret_cast<const float *>(tensor.Data().get());
  std::cout << "output data is:";
  for (int i = 0; i < tensor.ElementNum() && i <= 50; i++) {
    std::cout << out_data[i] << " ";
  }
  std::cout << std::endl;
}
```

## Memory Release

If the inference process of MindSpore Lite is complete, release the created `Model`.

```c++
// Delete model.
delete model;
```
