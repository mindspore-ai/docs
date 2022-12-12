# 体验C++极简云侧推理Demo

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/quick_start/cloud_infer/quick_start_cpp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

本教程提供了MindSpore Lite执行云侧推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了C++进行云侧推理的基本流程，用户能够快速了解MindSpore Lite执行推理相关API的使用。相关样例代码放置在[mindspore/lite/examples/cloud_infer/quick_start_cpp](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/quick_start_cpp)目录。

MindSpore Lite云侧推理仅支持在Linux环境部署运行。支持Ascend310、Ascend310P、Nvidia GPU和CPU硬件后端。

如需体验MindSpore Lite端侧推理流程，请参考文档[体验C++极简端侧推理Demo](https://www.mindspore.cn/lite/docs/zh-CN/master/quick_start/quick_start_cpp.html)。

使用MindSpore Lite执行推理主要包括以下步骤：

1. 模型读取：通过MindSpore导出MindIR模型，或者由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)转换获得MindIR模型。
2. 创建配置上下文：创建配置上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)，保存需要的一些基本配置参数，用于指导模型编译和模型执行。
3. 模型加载与编译：执行推理之前，需要调用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)的[Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build-3)接口进行模型加载和模型编译，并将上一步得到的Context配置到Model中。模型加载阶段会耗费较多时间，所以建议Model创建一次，编译一次，多次推理。
4. 输入数据：模型执行之前需要填充输入数据。
5. 执行推理：使用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)的[Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict)接口进行模型推理。

![img](../../images/lite_runtime.png)

> 如需查看MindSpore Lite高级用法，请参考[使用Runtime执行云侧推理（C++）](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/runtime_cpp_extendrt.html)。

## 构建与运行

MindSpore Lite云侧统一推理包仅支持在Linux环境部署运行。

- 环境要求

    - 系统环境：Linux
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- 编译构建

  在`mindspore/lite/examples/cloud_infer/quick_start_cpp`目录下执行[build.sh脚本](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/cloud_infer/quick_start_cpp/build.sh)，将自动下载MindSpore Lite推理框架库以及文模型文件并编译Demo。

  ```bash
  bash build.sh
  ```

  > 若使用该build脚本下载MindSpore Lite推理框架失败，请手动从[官网](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)下载Ascend、Nvidia GPU、CPU三合一的MindSpore Lite 模型推理包`mindspore-lite-{version}-linux-{arch}.tar.gz`，并存放到`mindspore/lite/examples/cloud_infer/quick_start_cpp`目录。
  >
  > 若MobileNetV2模型下载失败，请手动下载相关模型文件[mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir)，并将其拷贝到`mindspore/lite/examples/cloud_infer/quick_start_cpp/model`目录。
  >
  > 通过手动下载并且将文件放到指定位置后，再次执行build.sh脚本完成编译构建。

- 执行推理

  设置环境变量`LITE_HOME`为MindSpore Lite tar包解压路径，并设置环境变量`LD_LIBRARY_PATH`：

  ```bash
  export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
  ```

  编译构建后，进入`mindspore/lite/examples/cloud_infer/quick_start_cpp/build`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  ./mindspore_quick_start_cpp ../model/mobilenetv2.mindir
  ```

  执行完成后将能得到如下结果，打印输出Tensor的名称、输出Tensor的大小，输出Tensor的数量以及前50个数据：

  ```text
  tensor name is:Default/head-MobileNetV2Head/Softmax-op204 tensor size is:4000 tensor elements num is:1000
  output data is:5.07155e-05 0.00048712 0.000312549 0.00035624 0.0002022 8.58958e-05 0.000187147 0.000365937 0.000281044 0.000255672 0.00108948 0.00390996 0.00230398 0.00128984 0.00307477 0.00147607 0.00106759 0.000589853 0.000848115 0.00143693 0.000685777 0.00219331 0.00160639 0.00215123 0.000444315 0.000151986 0.000317552 0.00053971 0.00018703 0.000643944 0.000218269 0.000931556 0.000127084 0.000544278 0.000887942 0.000303909 0.000273875 0.00035335 0.00229062 0.000453207 0.0011987 0.000621194 0.000628335 0.000838564 0.000611029 0.000372603 0.00147742 0.000270685 8.29869e-05 0.000116974 0.000876237
  ```

## 配置CMake

以下是通过CMake集成`libmindspore-lite.so`动态库时的示例代码。读取环境变量`LITE_HOME`以获取MindSpore Lite tar包解压后的头文件和库文件目录。

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

## 创建配置上下文

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

## 模型创建加载与编译

模型加载与编译可以调用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)的复合[Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build)接口，从文件加载、编译得到运行时的模型。

```c++
mindspore::Model model;
// Build model
auto build_ret = model.Build(model_path, mindspore::kMindIR, context);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model error " << build_ret << std::endl;
  return -1;
}
```

## 模型推理

模型推理主要包括输入数据、执行推理、获得输出等步骤，其中本示例中的输入数据是通过随机数据构造生成，最后将执行推理后的输出结果打印出来。

```c++
// Get Input
auto inputs = model.GetInputs();
// Generate random data as input data.
auto ret = GenerateInputDataWithRandom(inputs);
if (ret != mindspore::kSuccess) {
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
```
