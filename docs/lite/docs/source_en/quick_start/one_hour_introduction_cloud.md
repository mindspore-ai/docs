# Quick Start to Cloud-Side Inference

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/quick_start/one_hour_introduction_cloud.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This article introduces you to the basic functions and usage of MindSpore Lite by using MindSpore Lite to perform cloud-side inference as an example.

MindSpore Lite cloud-side inference is supported to run in Linux environment deployment only. Ascend310, Ascend310P, Nvidia GPU and CPU hardware backends are supported.

Before starting using MindSpore Lite in this chapter, users should have a Linux (e.g. Ubuntu/CentOS/EulerOS) environment ready to operate the verification.

To experience the MindSpore Lite device-side inference process, please refer to the document [Quick Start to Device-Side Inference](https://www.mindspore.cn/lite/docs/en/master/quick_start/one_hour_introduction.html).

We will demonstrate how to use MindSpore Lite distributions for integrated development and write your own inference programs, taking MindSpore Lite C++ interface for integration as an example. For detailed usage of MindSpore Lite C++ interface, users can refer to [Cloud-Side inference with C++ Interface](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/runtime_cpp.html).

In addition, users can use Python interface and Java interface of MindSpore Lite for integration. For details, please refer to [Cloud-side inference by using Python interface](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/runtime_python.html) and [Cloud-side inference by using Java interface](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/runtime_java.html).

## Preparation

1. Environment requirements
    - System environment: Linux x86_64, Ubuntu 18.04.02LTS recommended
    - C++ compilation dependencies
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
        - [CMake](https://cmake.org/download/) >= 3.12

2. Download distributions

    Users can download the MindSpore Lite cloud-side inference package `mindspore-lite-{ version}-linux-{arch}.tar.gz` on the [download page](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) of MindSpore official website, `{arch}` for `x86` or `aarch64`. `x86` version supports Ascend, Nvidia GPU, CPU three hardware backends, `aarch64` only supports Ascend and CPU hardware backends.

    ```text
    mindspore-lite-{version}-linux-x64
    ├── runtime
    │   ├── include                        # API header files for MindSpore Lite integrated development
    │   ├── lib
    │   │   ├── libascend_ge_plugin.so     # Ascend Hardware Backend Remote Mode Plugin
    │   │   ├── libascend_kernel_plugin.so # Ascend Hardware Backend Plugin
    │   │   ├── libdvpp_utils.so           # Ascend Hardware Backend DVPP Plugin
    │   │   ├── libminddata-lite.a         # Image processing static library
    │   │   ├── libminddata-lite.so        # Image processing dynamic library
    │   │   ├── libmindspore_core.so       # Dynamic library for MindSpore Lite inference framework
    │   │   ├── libmindspore_glog.so.0     # MindSpore Lite Logging Dynamic Library
    │   │   ├── libmindspore-lite-jni.so   # JNI dynamic library for MindSpore Lite inference framework
    │   │   ├── libmindspore-lite.so       # Dynamic library for MindSpore Lite inference framework
    │   │   ├── libmsplugin-ge-litert.so
    │   │   ├── libruntime_convert_plugin.so
    │   │   ├── libtensorrt_plugin.so      # Nvidia GPU Hardware Backend Plugin
    │   │   ├── libtransformer-shared.so   # Transformer Dynamic Library
    │   │   └── mindspore-lite-java.jar    # MindSpore Lite inference framework jar package
    │   └── third_party
    │       └── libjpeg-turbo
    └── tools
        ├── benchmark       # Benchmark Test Tools Catalogue
        └── converter       # Model Converter Catalogue
    ```

3. Obtain model

    MindSpore Lite cloud-side inference currently only supports MindIR model format of MindSpore. You can export MindIR model by MindSpore or get MindIR model by [model converter](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html) to convert models in Tensorflow, Onnx, Caffe.

    The model file [mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir) can be downloaded as a sample model.

4. Obtain sample

    The sample code of this section is put in the directory [mindspore/lite/examples/cloud_infer/quick_start_cpp](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/quick_start_cpp).

    ```text
    quick_start_cpp
    ├── CMakeLists.txt
    ├── main.cc
    ├── build                           # Temporary build directory
    └── model
        └── mobilenetv2.mindir          # Model files
    ```

## Environment Variables

**To ensure that the script will work properly, environment variables need to be set before building and executing the inference.**

### MindSpore Lite Environment Variables

After unzipping the MindSpore Lite cloud-side inference package, set the `LITE_HOME` environment variable to the path of the unzipping, e.g.

```bash
export LITE_HOME=$some_path/mindpsore-lite-2.0.0-linux-x64
```

Set the environment variable `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
```

If you need to use the `convert_lite` or `benchmark` tools, you need to set the environment variable `PATH`.

```bash
export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
```

### Ascend Hardware Backend Environment Variables

1. Verify the run package installation path

    If you use the root user to complete the run package installation, the default path is '/usr/local/Ascend', and the default installation path for non-root users is '/home/HwHiAiUser/Ascend'.

    Taking the path of the root user as an example, set the environment variables as follows:

    ```bash
    export ASCEND_HOME=/usr/local/Ascend  # the root directory of run package
    ```

2. Distinguish run package versions

    The run package is divided into 2 versions, distinguished by whether the 'ascend-toolkit' folder is set in the installation directory.

    If the 'ascend-toolkit' folder exists, set the environment variables as follows:

    ```bash
    export ASCEND_HOME=/usr/local/Ascend
    export PATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/bin:${ASCEND_HOME}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
    export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH}
    export ASCEND_OPP_PATH=${ASCEND_HOME}/ascend-toolkit/latest/opp
    export ASCEND_AICPU_PATH=${ASCEND_HOME}/ascend-toolkit/latest/
    export PYTHONPATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/python/site-packages:${PYTHONPATH}
    export TOOLCHAIN_HOME=${ASCEND_HOME}/ascend-toolkit/latest/toolkit
    ```

    If not exist, set the environment variables as follows:

    ```bash
    export ASCEND_HOME=/usr/local/Ascend
    export PATH=${ASCEND_HOME}/latest/compiler/bin:${ASCEND_HOME}/latest/compiler/ccec_compiler/bin:${PATH}
    export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/latest/lib64:${LD_LIBRARY_PATH}
    export ASCEND_OPP_PATH=${ASCEND_HOME}/latest/opp
    export ASCEND_AICPU_PATH=${ASCEND_HOME}/latest
    export PYTHONPATH=${ASCEND_HOME}/latest/compiler/python/site-packages:${PYTHONPATH}
    export TOOLCHAIN_HOME=${ASCEND_HOME}/latest/toolkit
    ```

### Nvidia GPU Hardware Backend Environment Variables

When the hardware backend is an Nvidia GPU, inference relies on cuda and TensorRT, and users need to install cuda and TensorRT first.

The following is an example of cuda11.1 and TensorRT8.2.5.1. Users need to set the environment variables according to the actual installation path.

```bash
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TENSORRT_PATH=/usr/local/TensorRT-8.2.5.1
export PATH=$TENSORRT_PATH/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_PATH/lib:$LD_LIBRARY_PATH
```

### Setting Host-side Logging Level

The Host logging level defaults to `WARNING`.

```bash
export GLOG_v=2 # 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
```

## Integration Inference

We will demonstrate how to use MindSpore Lite distributions for integrated development and write your own inference programs, using MindSpore Lite C++ interface for integration as an example.

Before integration, users can also directly use the [benchmark tool (benchmark)](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/benchmark_tool.html) distributed with the distribution to perform inference tests.

### Configuring CMake

Users need to integrate the `mindspore-lite` library file inside the distribution and perform model inference through the API interface declared in the MindSpore Lite header file.

The following is sample code when integrating the `libmindspore-lite.so` dynamic library via CMake. The environment variable `LITE_HOME` is read to get the unpacked header and library file directories of MindSpore Lite tar package.

```cmake
cmake_minimum_required(VERSION 3.14)
project(QuickStartCpp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

if(DEFINED ENV{LITE_HOME})
    set(LITE_HOME $ENV{LITE_HOME})
endif()

# Add directory to include search path
include_directories(${LITE_HOME}/runtime)
# Add directory to linker search path
link_directories(${LITE_HOME}/runtime/lib)
link_directories(${LITE_HOME}/tools/converter/lib)

file(GLOB_RECURSE QUICK_START_CXX ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_executable(mindspore_quick_start_cpp ${QUICK_START_CXX})

target_link_libraries(mindspore_quick_start_cpp mindspore-lite pthread dl)
```

### Writing Code

The code in `main.cc` is shown below:

```cpp
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <cstring>
#include <memory>
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/api/types.h"

template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  std::mt19937 random_engine;
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&distribution, &random_engine]() { return static_cast<T>(distribution(random_engine)); });
}

int GenerateInputDataWithRandom(std::vector<mindspore::MSTensor> inputs) {
  for (auto tensor : inputs) {
    auto input_data = tensor.MutableData();
    if (input_data == nullptr) {
      std::cerr << "MallocData for inTensor failed." << std::endl;
      return -1;
    }
    GenerateRandomData<float>(tensor.DataSize(), input_data, std::uniform_real_distribution<float>(0.1f, 1.0f));
  }
  return 0;
}

int QuickStart(int argc, const char **argv) {
  if (argc < 2) {
    std::cerr << "Model file must be provided.\n";
    return -1;
  }
  // Read model file.
  std::string model_path = argv[1];
  if (model_path.empty()) {
    std::cerr << "Model path " << model_path << " is invalid.";
    return -1;
  }

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

  mindspore::Model model;
  // Build model
  auto build_ret = model.Build(model_path, mindspore::kMindIR, context);
  if (build_ret != mindspore::kSuccess) {
    std::cerr << "Build model error " << build_ret << std::endl;
    return -1;
  }

  // Get Input
  auto inputs = model.GetInputs();
  // Generate random data as input data.
  if (GenerateInputDataWithRandom(inputs) != 0) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
    return -1;
  }

  // Model Predict
  std::vector<mindspore::MSTensor> outputs;
  auto predict_ret = model.Predict(inputs, &outputs);
  if (predict_ret != mindspore::kSuccess) {
    std::cerr << "Predict error " << predict_ret << std::endl;
    return -1;
  }

  // Print Output Tensor Data.
  constexpr int kNumPrintOfOutData = 50;
  for (auto &tensor : outputs) {
    std::cout << "tensor name is:" << tensor.Name() << " tensor size is:" << tensor.DataSize()
              << " tensor elements num is:" << tensor.ElementNum() << std::endl;
    auto out_data = reinterpret_cast<const float *>(tensor.Data().get());
    std::cout << "output data is:";
    for (int i = 0; i < tensor.ElementNum() && i <= kNumPrintOfOutData; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}

int main(int argc, const char **argv) { return QuickStart(argc, argv); }
```

The code function is parsed as follows:

1. Initialize the Context configuration

    Context holds the relevant configurations needed for model inference, including operator preferences, number of threads, automatic concurrency, and other configurations related to the inference processor.
    For more details about Context, please refer to [API interface description](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore.html#context) of Context.
    When loading the model in MindSpore Lite, an object of class `Context` must be provided, so in this example, an object `context` of class `Context` is first requested.

    ```cpp
    auto context = std::make_shared<mindspore::Context>();
    ```

    Next, get the device management list of the `context` object through the `Context::MutableDeviceInfo` interface.

    ```cpp
    auto &device_list = context->MutableDeviceInfo();
    ```

    In this example, since the CPU is used for inference, an object `device_info` of class `CPUDeviceInfo` needs to be requested.

    ```cpp
    auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
    ```

    Since the default CPU settings are used, there is no need to do any settings for the `device_info` object and it is directly added to the device management list of `context`.

    ```cpp
    device_list.push_back(device_info);
    ```

2. Load models

    First create the object `model` of a `Model` class, and the `Model` class defines the model in MindSpore for computational graph management.
    For a detailed description of the `Model` class, please refer to the [API documentation](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore.html#model).

    ```cpp
    mindspore::Model model;
    ```

    Then call the `Build` interface to pass in the model and compile it to a running state on the device.

    ```cpp
    auto build_ret = model.Build(model_path, mindspore::kMindIR, context);
    ```

3. Pass in data

    Before performing model inference, you need to set the input data for inference.
    In this example, all the input tensor of the model is obtained through the `Model.GetInputs` interface. The format of the individual tensor is `MSTensor`.
    For a detailed description of the `MSTensor` tensor, please refer to the [API description](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore.html#mstensor) of `MSTensor`.

    ```cpp
    auto inputs = model.GetInputs();
    ```

    The `MutableData` interface of the tensor can get the data memory pointer of the tensor, and the `DataSize` interface of the tensor can get the data byte length of the tensor. The data type of the tensor can be obtained through the `DataType` interface of the tensor, and users can do different processing according to the data format of their models.

    ```cpp
    auto input_data = tensor.MutableData();
    ```

    Next, the data on which we want to perform inference is passed inside the tensor via a data pointer.
    In this case we pass in floating point data randomly generated from 0.1 to 1 and the data is evenly distributed.
    In practical inference, after reading the actual data such as images or audio, the user needs to perform algorithm-specific pre-processing operations and pass the processed data into the model.

    ```cpp
    template <typename T, typename Distribution>
    void GenerateRandomData(int size, void *data, Distribution distribution) {
      std::mt19937 random_engine;
      int elements_num = size / sizeof(T);
      (void)std::generate_n(static_cast<T *>(data), elements_num,
                            [&distribution, &random_engine]() { return static_cast<T>(distribution(random_engine)); });
    }

    int GenerateInputDataWithRandom(std::vector<mindspore::MSTensor> inputs) {
      for (auto tensor : inputs) {
        auto input_data = tensor.MutableData();
        if (input_data == nullptr) {
          std::cerr << "MallocData for inTensor failed." << std::endl;
          return -1;
        }
        GenerateRandomData<float>(tensor.DataSize(), input_data, std::uniform_real_distribution<float>(0.1f, 1.0f));
      }
      return 0;
    }

      // Get Input
      auto inputs = model.GetInputs();
      // Generate random data as input data.
      if (GenerateInputDataWithRandom(inputs) != 0) {
        std::cerr << "Generate Random Input Data failed." << std::endl;
        return -1;
      }
    ```

4. Execute inference

    First, an array `outputs` is requested to hold the output tensor of the model inference, and then the model inference interface `Predict` is called with the input tensor and output tensor as its parameters.
    After a successful inference, the output tensor is stored in `outputs`.

    ```cpp
    std::vector<MSTensor> outputs;
    auto status = model.Predict(inputs, &outputs);
    ```

5. Obtain inference results

    The data pointer to the output tensor is obtained via `Data`.
    In this case, it is strongly converted to a floating point pointer, and the user can convert the corresponding type according to the data type of model, or get the data type through the `DataType` interface of the tensor.

    ```cpp
    auto out_data = reinterpret_cast<float *>(tensor.Data().get());
    ```

    In this example, the inference output is printed directly.

    ```cpp
    for (int i = 0; i < tensor.ElementNum() && i <= kNumPrintOfOutData; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
    ```

6. Release the model object

    Model destructions will release model-related resources.

### Compiling

Set the environment variables as described in the Environment Variables section. Then compile the program as follows.

```bash
mkdir build && cd build
cmake ../
make
```

After successful compilation, you can get the `quick_start_cpp` executable in the `build` directory.

### Running the Inference Program

```bash
./mindspore_quick_start_cpp ../model/mobilenetv2.mindir
```

After execution, the following results will be obtained, printing the name of the output Tensor, the size of the output Tensor, the number of the output Tensor and the first 50 data:

```text
tensor name is:Default/head-MobileNetV2Head/Softmax-op204 tensor size is:4000 tensor elements num is:1000
output data is:5.07155e-05 0.00048712 0.000312549 0.00035624 0.0002022 8.58958e-05 0.000187147 0.000365937 0.000281044 0.000255672 0.00108948 0.00390996 0.00230398 0.00128984 0.00307477 0.00147607 0.00106759 0.000589853 0.000848115 0.00143693 0.000685777 0.00219331 0.00160639 0.00215123 0.000444315 0.000151986 0.000317552 0.00053971 0.00018703 0.000643944 0.000218269 0.000931556 0.000127084 0.000544278 0.000887942 0.000303909 0.000273875 0.00035335 0.00229062 0.000453207 0.0011987 0.000621194 0.000628335 0.000838564 0.000611029 0.000372603 0.00147742 0.000270685 8.29869e-05 0.000116974 0.000876237
```
