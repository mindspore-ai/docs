# 体验C++极简推理Demo

`Linux` `Windows` `X86` `C++` `全流程` `推理应用` `数据准备` `初级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/quick_start/quick_start_server_inference_cpp.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

本教程提供了MindSpore Lite执行并发推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了C++进行并发推理的基本流程，用户能够快速了解MindSpore Lite执行并发推理相关API的使用。本教程通过随机生成的数据作为输入数据，执行MobileNetV2模型的推理，打印获得输出数据。相关代码放置在[mindspore/lite/examples/quick_start_server_inference_cpp](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_server_inference_cpp)目录。

使用MindSpore Lite执行推理主要包括以下步骤：

1. 模型加载(可选)：从文件系统中读取由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)转换得到的`.ms`模型。
2. 创建配置选项：创建并发推理配置选项[RunnerConfig](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#runnerconfig)，保存需要的一些基本配置参数，用于执行并发推理的初始化。
3. 初始化：在执行并发推理前，需要调用[ModelParallelRunner](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#modelparallelrunner)的[init](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#init)接口进行并发推理的初始化，主要进行模型读取，创建并发，以及子图切分、算子选型调度。这部分会耗费较多时间，所以建议[init](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#init)初始化一次，多次执行并发推理。
4. 输入数据：模型执行之前需要向`输入Tensor`中填充数据。
5. 执行推理：使用[ModelParallelRunner](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#modelparallelrunner)的[Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict)接口进行并发推理。
6. 获得输出：模型执行结束之后，可以通过`输出Tensor`得到推理结果。
7. 释放内存：无需使用MindSpore Lite并发推理框架时，需要释放已创建的[ModelParallelRunner](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#modelparallelrunner)。

![img](../images/server_inference.png)

## 构建与运行

### Linux X86

- 环境要求

    - 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- 编译构建

  在`mindspore/lite/examples/quick_start_server_inference_cpp`目录下执行[build脚本](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_server_inference_cpp/build.sh)，将自动下载MindSpore Lite推理框架库以及文模型文件并编译Demo。

  ```bash
  bash build.sh
  ```

  > 若使用该build脚本下载MindSpore Lite推理框架失败，请手动下载硬件平台为CPU、操作系统为Ubuntu-x64的MindSpore Lite 模型推理框架[mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)，将解压后`runtime/lib`目录下的`libmindspore-lite.so`文件拷贝到`mindspore/lite/examples/quick_start_server_inference_cpp/lib`目录、`runtime/include`目录里的文件拷贝到`mindspore/lite/examples/quick_start_server_inference_cpp/include`目录下。
  >
  > 若MobileNetV2模型下载失败，请手动下载相关模型文件[mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms)，并将其拷贝到`mindspore/lite/examples/quick_start_server_inference_cpp/model`目录。
  >
  > 通过手动下载并且将文件放到指定位置后，需要再次执行build.sh脚本才能完成编译构建。

- 执行推理

  编译构建后，进入`mindspore/lite/examples/quick_start_server_inference_cpp/build`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  ./mindspore_quick_start_cpp ../model/mobilenetv2.ms
  ```

  执行完成后将能得到如下结果，打印输出Tensor的名称、输出Tensor的大小，输出Tensor的数量以及前50个数据：

  ```text
  tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
  output data is:1.74225e-05 1.15919e-05 2.02728e-05 0.000106485 0.000124295 0.00140576 0.000185107 0.000762011 1.50996e-05 5.91942e-06 6.61469e-06 3.72883e-06 4.30761e-06 2.38897e-06 1.5163e-05 0.000192663 1.03767e-05 1.31953e-05 6.69638e-06 3.17411e-05 4.00895e-06 9.9641e-06 3.85127e-06 6.25101e-06 9.08853e-06 1.25043e-05 1.71761e-05 4.92751e-06 2.87637e-05 7.46446e-06 1.39375e-05 2.18824e-05 1.08861e-05 2.5007e-06 3.49876e-05 0.000384547 5.70778e-06 1.28909e-05 1.11038e-05 3.53906e-06 5.478e-06 9.76608e-06 5.32172e-06 1.10386e-05 5.35474e-06 1.35796e-05 7.12652e-06 3.10017e-05 4.34154e-06 7.89482e-05 1.79441e-05
  ```

## 初始化

```c++
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

## 执行推理

模型推理主要包括输入数据、执行推理、获得输出等步骤，其中本示例中的输入数据是通过随机数据构造生成，最后将执行推理后的输出结果打印出来。

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

## 内存释放

无需使用MindSpore Lite推理框架时，需要释放已经创建的`ModelParallelRunner`。

```c++
delete model_runner;
```
