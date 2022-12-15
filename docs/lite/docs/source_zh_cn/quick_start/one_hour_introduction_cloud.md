# 云侧推理快速入门

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/quick_start/one_hour_introduction_cloud.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

本文通过使用MindSpore Lite执行云侧推理为例，向大家介绍MindSpore Lite的基础功能和用法。

MindSpore Lite云侧推理仅支持在Linux环境部署运行。支持Ascend310、Ascend310P、Nvidia GPU和CPU硬件后端。
在开始本章的MindSpore Lite使用之旅之前，用户需拥有一个Linux（如Ubuntu/CentOS/EulerOS）的环境，以便随时操作验证。

如需体验MindSpore Lite端侧推理流程，请参考文档[端侧推理快速入门](https://www.mindspore.cn/lite/docs/zh-CN/master/quick_start/one_hour_introduction.html)。

我们将以使用MindSpore Lite的C++接口进行集成为例，演示如何使用MindSpore Lite的发布件，进行集成开发，编写自己的推理程序。MindSpore Lite的C++接口的详细用法用户可参考[使用C++接口进行云侧推理](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/runtime_cpp.html)。

另外，用户可以使用MindSpore Lite的Python接口Java接口进行集成。详情可参考[使用Python接口进行云侧推理](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/runtime_python.html)和[使用Java接口进行云侧推理](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/runtime_java.html)。

## 准备工作

1. 环境要求
    - 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
    - C++编译依赖
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
        - [CMake](https://cmake.org/download/) >= 3.12

2. 下载发布件

    用户可在MindSpore官网[下载页面](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)下载MindSpore Lite云侧推理包`mindspore-lite-{version}-linux-{arch}.tar.gz`，`{arch}`为`x86`或者`aarch64`，`x86`版本支持Ascend、Nvidia GPU、CPU三个硬件后端，`aarch64`仅支持Ascend、CPU硬件后端。

    ```text
    mindspore-lite-{version}-linux-x64
    ├── runtime
    │   ├── include                        # MindSpore Lite集成开发的API头文件
    │   ├── lib
    │   │   ├── libascend_ge_plugin.so     # Ascend硬件后端拉远模式插件
    │   │   ├── libascend_kernel_plugin.so # Ascend硬件后端插件
    │   │   ├── libdvpp_utils.so           # Ascend硬件后端DVPP插件
    │   │   ├── libminddata-lite.a         # 图像处理静态库
    │   │   ├── libminddata-lite.so        # 图像处理动态库
    │   │   ├── libmindspore_core.so       # MindSpore Lite推理框架的动态库
    │   │   ├── libmindspore_glog.so.0     # MindSpore Lite日志动态库
    │   │   ├── libmindspore-lite-jni.so   # MindSpore Lite推理框架的JNI动态库
    │   │   ├── libmindspore-lite.so       # MindSpore Lite推理框架的动态库
    │   │   ├── libmsplugin-ge-litert.so
    │   │   ├── libruntime_convert_plugin.so
    │   │   ├── libtensorrt_plugin.so      # Nvidia GPU硬件后端插件
    │   │   ├── libtransformer-shared.so   # Transformer动态库
    │   │   └── mindspore-lite-java.jar    # MindSpore Lite推理框架jar包
    │   └── third_party
    │       └── libjpeg-turbo
    └── tools
        ├── benchmark       # 基准测试工具目录
        └── converter       # 模型转换工具目录
    ```

3. 获取模型

    MindSpore Lite云侧推理当前仅支持MindSpore的MindIR模型格式，可以通过MindSpore导出MindIR模型，或者由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)转换Tensorflow、Onnx、Caffe等格式的模型获得MindIR模型。

    可下载模型文件[mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir)作为样例模型。

4. 获取样例

    本节样例代码放置在[mindspore/lite/examples/cloud_infer/quick_start_cpp](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/quick_start_cpp)目录。

    ```text
    quick_start_cpp
    ├── CMakeLists.txt
    ├── main.cc
    ├── build                           # 临时的构建目录
    └── model
        └── mobilenetv2.mindir          # 模型文件
    ```

## 环境变量

**为了确保脚本能够正常地运行，需要在构建和执行推理前设置环境变量。**

### MindSpore Lite环境变量

MindSpore Lite云侧推理包解压缩后，设置`LITE_HOME`环境变量为解压缩的路径，比如：

```bash
export LITE_HOME=$some_path/mindpsore-lite-2.0.0-linux-x64
```

设置环境变量`LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
```

如果需要使用`convert_lite`或者`benchmark`工具，则需要设置环境变量`PATH`。

```bash
export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH
```

### Ascend硬件后端环境变量

1. 确认run包安装路径

    若使用root用户完成run包安装，默认路径为'/usr/local/Ascend'，非root用户的默认安装路径为'/home/HwHiAiUser/Ascend'。

    以root用户的路径为例，设置环境变量如下：

    ```bash
    export ASCEND_HOME=/usr/local/Ascend  # the root directory of run package
    ```

2. 区分run包版本

    run包分为2个版本，用安装目录下是否存在'ascend-toolkit'文件夹进行区分。

    如果存在'ascend-toolkit'文件夹，设置环境变量如下：

    ```bash
    export ASCEND_HOME=/usr/local/Ascend
    export PATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/bin:${ASCEND_HOME}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
    export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH}
    export ASCEND_OPP_PATH=${ASCEND_HOME}/ascend-toolkit/latest/opp
    export ASCEND_AICPU_PATH=${ASCEND_HOME}/ascend-toolkit/latest/
    export PYTHONPATH=${ASCEND_HOME}/ascend-toolkit/latest/compiler/python/site-packages:${PYTHONPATH}
    export TOOLCHAIN_HOME=${ASCEND_HOME}/ascend-toolkit/latest/toolkit
    ```

    若不存在，设置环境变量为：

    ```bash
    export ASCEND_HOME=/usr/local/Ascend
    export PATH=${ASCEND_HOME}/latest/compiler/bin:${ASCEND_HOME}/latest/compiler/ccec_compiler/bin:${PATH}
    export LD_LIBRARY_PATH=${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/latest/lib64:${LD_LIBRARY_PATH}
    export ASCEND_OPP_PATH=${ASCEND_HOME}/latest/opp
    export ASCEND_AICPU_PATH=${ASCEND_HOME}/latest
    export PYTHONPATH=${ASCEND_HOME}/latest/compiler/python/site-packages:${PYTHONPATH}
    export TOOLCHAIN_HOME=${ASCEND_HOME}/latest/toolkit
    ```

### Nvidia GPU硬件后端环境变量

硬件后端为Nvidia GPU时，推理依赖cuda和TensorRT，用户需要先安装cuda和TensorRT。

以下以cuda11.1和TensorRT8.2.5.1为例，用户需要根据实际安装路径设置环境变量。

```bash
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TENSORRT_PATH=/usr/local/TensorRT-8.2.5.1
export PATH=$TENSORRT_PATH/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_PATH/lib:$LD_LIBRARY_PATH
```

### 设置Host侧日志级别

Host日志级别默认为`WARNING`。

```bash
export GLOG_v=2 # 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
```

## 集成推理

我们将以使用MindSpore Lite的C++接口进行集成为例，演示如何使用MindSpore Lite的发布件，进行集成开发，编写自己的推理程序。

在进行集成前，用户也可以直接使用随发布件发布的[基准测试工具（benchmark）](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/benchmark_tool.html)来进行推理测试。

### 配置CMake

用户需要集成发布件内的`mindspore-lite`库文件，并通过MindSpore Lite头文件中声明的API接口，来进行模型推理。

以下是通过CMake集成`libmindspore-lite.so`动态库时的示例代码。通过读取环境变量`LITE_HOME`以获取MindSpore Lite tar包解压后的头文件和库文件目录。

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

### 编写代码

`main.cc`中代码如下所示：

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

代码功能解析如下：

1. 初始化Context配置

    Context保存了模型推理时所需的相关配置，包括算子偏好、线程数、自动并发以及推理处理器相关的其他配置。
    关于Context的详细说明，请参考Context的[API接口说明](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)。
    在MindSpore Lite加载模型时，必须提供一个`Context`类的对象，所以在本例中，首先申请了一个`Context`类的对象`context`。

    ```cpp
    auto context = std::make_shared<mindspore::Context>();
    ```

    接着，通过`Context::MutableDeviceInfo`接口，得到`context`对象的设备管理列表。

    ```cpp
    auto &device_list = context->MutableDeviceInfo();
    ```

    在本例中，由于使用CPU进行推理，故需申请一个`CPUDeviceInfo`类的对象`device_info`。

    ```cpp
    auto device_info = std::make_shared<mindspore::CPUDeviceInfo>();
    ```

    因为采用了CPU的默认设置，所以不需对`device_info`对象做任何设置，直接添加到`context`的设备管理列表。

    ```cpp
    device_list.push_back(device_info);
    ```

2. 加载模型

    首先创建一个`Model`类对象`model`，`Model`类定义了MindSpore中的模型，用于计算图管理。
    关于`Model`类的详细说明，可参考[API文档](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)。

    ```cpp
    mindspore::Model model;
    ```

    接着调用`Build`接口传入模型，将模型编译至可在设备上运行的状态。

    ```cpp
    auto build_ret = model.Build(model_path, mindspore::kMindIR, context);
    ```

3. 传入数据

    在执行模型推理前，需要设置推理的输入数据。
    此例，通过`Model.GetInputs`接口，获取模型的所有输入张量。单个张量的格式为`MSTensor`。
    关于`MSTensor`张量的详细说明，请参考`MSTensor`的[API说明](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    ```cpp
    auto inputs = model.GetInputs();
    ```

    通过张量的`MutableData`接口可以获取张量的数据内存指针，通过张量的`DataSize`接口可以获取张量的数据字节长度。可通过张量的`DataType`接口得到该张量的数据类型，用户可根据自己模型的数据格式做不同处理。

    ```cpp
    auto input_data = tensor.MutableData();
    ```

    接着，通过数据指针，将我们要推理的数据传入张量内部。
    在本例中我们传入的是随机生成的0.1至1的浮点数据，且数据呈平均分布。
    在实际的推理中，用户在读取图片或音频等实际数据后，需进行算法特定的预处理操作，并将处理后的数据传入模型。

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

4. 执行推理

    首先申请一个放置模型推理输出张量的数组`outputs`，然后调用模型推理接口`Predict`，将输入张量和输出张量作它的参数。
    在推理成功后，输出张量被保存在`outputs`内。

    ```cpp
    std::vector<MSTensor> outputs;
    auto status = model.Predict(inputs, &outputs);
    ```

5. 获取推理结果

    通过`Data`得到输出张量的数据指针。
    本例中，将它强转为浮点指针，用户可以根据自己模型的数据类型进行对应类型的转换，也可通过张量的`DataType`接口得到数据类型。

    ```cpp
    auto out_data = reinterpret_cast<float *>(tensor.Data().get());
    ```

    在本例中，直接打印推理输出结果。

    ```cpp
    for (int i = 0; i < tensor.ElementNum() && i <= kNumPrintOfOutData; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
    ```

6. 释放model对象

    Model析构时将释放模型相关资源。

### 编译

按照环境变量一节所述，设置环境变量。接着按如下方式编译程序。

```bash
mkdir build && cd build
cmake ../
make
```

在编译成功后，可以在`build`目录下得到`quick_start_cpp`可执行程序。

### 运行推理程序

```bash
./mindspore_quick_start_cpp ../model/mobilenetv2.mindir
```

执行完成后将能得到如下结果，打印输出Tensor的名称、输出Tensor的大小，输出Tensor的数量以及前50个数据：

```text
tensor name is:Default/head-MobileNetV2Head/Softmax-op204 tensor size is:4000 tensor elements num is:1000
output data is:5.07155e-05 0.00048712 0.000312549 0.00035624 0.0002022 8.58958e-05 0.000187147 0.000365937 0.000281044 0.000255672 0.00108948 0.00390996 0.00230398 0.00128984 0.00307477 0.00147607 0.00106759 0.000589853 0.000848115 0.00143693 0.000685777 0.00219331 0.00160639 0.00215123 0.000444315 0.000151986 0.000317552 0.00053971 0.00018703 0.000643944 0.000218269 0.000931556 0.000127084 0.000544278 0.000887942 0.000303909 0.000273875 0.00035335 0.00229062 0.000453207 0.0011987 0.000621194 0.000628335 0.000838564 0.000611029 0.000372603 0.00147742 0.000270685 8.29869e-05 0.000116974 0.000876237
```
