# 体验MindSpore Lite C++ 极简Demo

`Linux` `X86` `C++` `全流程` `推理应用` `数据准备` `初级`

<!-- TOC -->

- [体验MindSpore Lite C++ 极简Demo](#体验mindspore-lite-c-极简demo)
    - [概述](#概述)
    - [构建与运行](#构建与运行)
    - [模型加载](#模型加载)
    - [模型编译](#模型编译)
    - [模型推理](#模型推理)
    - [内存释放](#内存释放)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_zh_cn/quick_start/quick_start_cpp.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

本教程提供了MindSpore Lite执行推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了C++进行端侧推理的基本流程，用户能够快速了解MindSpore Lite执行推理相关API的使用。本教程通过随机生成的数据作为数据数据，执行MobileNetV2模型的推理，打印获得输出数据。相关代码放置在[mindspore/lite/examples/quick_start_cpp](mindspore/lite/examples/quick_start_cpp)目录。

使用MindSpore Lite执行推理主要包括以下步骤：

1. 模型加载：从文件系统中读取由[模型转换工具](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/converter_tool.html)转换得到的`.ms`模型，通过[mindspore::lite::Model::Import](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#import)导入模型，进行模型解析，创建得到 `Model *`。
2. 创建配置上下文：创建配置上下文[Context](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#context)，保存会话所需的一些基本配置参数，用于指导图编译和图执行。
3. 创建会话：创建[LiteSession](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/session.html#litesession)会话，并将上一步得到的[Context](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#context)配置到会话中。
4. 图编译：执行推理之前，需要调用[LiteSession](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/session.html#litesession)的`CompileGraph`接口进行图编译。图编译阶段主要进行子图切分、算子选型调度等过程，该阶段会耗费较多时间，所以建议[LiteSession](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/session.html#litesession)创建一次，编译一次，多次推理。
5. 输入数据：图执行之前需要向`输入Tensor`中填充数据。
6. 执行推理：使用[LiteSession](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/session.html#litesession)的`RunGraph`进行模型推理。
7. 获得输出：图执行结束之后，可以通过`输出Tensor`得到推理结果。
8. 释放内存：无需使用MindSpore Lite推理框架时，需要释放已创建的[LiteSession](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/session.html#litesession)和[Model](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#model)。

![img](../images/lite_runtime.png)

> 如需查看MindSpore Lite高级用法，请参考[使用Runtime执行推理（C++）](https://www.mindspore.cn/tutorial/lite/zh-CN/master/quick_start/quick_start.html)。

## 构建与运行

- 环境要求
    - 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0
        - [Git](https://git-scm.com/downloads) >= 2.28.0

- 编译构建

  在`mindspore/lite/examples/quick_start_cpp`目录下执行build脚本，将能够自动下载相关文件并编译Demo。

  ```bash
  bash build.sh
  ```

  > 若MindSpore Lite推理框架下载失败，请手动下载硬件平台为CPU，操作系统为Ubuntu-x64的[MindSpore Lite 模型推理框架](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/downloads.html)，解压后将其拷贝对应到`mindspore/lite/examples/quick_start_cpp/lib`目录。
  >
  > 若mobilenetv2模型下载失败，请手动下载相关模型文件[mobilenetv2](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/1.1/mobilenetv2.ms)，并将其拷贝到`mindspore/lite/examples/quick_start_cpp/model`目录。

- 执行推理

  编译构建后，进入`mindspore/lite/examples/quick_start_cpp/build`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  ./mindspore_quick_start_cxx ../model/mobilenetv2.ms
  ```

  执行完成后将能得到如下结果，将打印输出Tensor的名称、输出Tensor的大小，输出Tensor的数量以及前50个数据：

  ```shell
  tensor name is:Default/Sigmoid-op204 tensor size is:2000 tensor elements num is:500
  output data is:3.31223e-05 1.99382e-05 3.01624e-05 0.000108345 1.19685e-05 4.25282e-06 0.00049955 0.000340809 0.00199094 0.000997094 0.00013585 1.57605e-05 4.34131e-05 1.56114e-05 0.000550819 2.9839e-05 4.70447e-06 6.91601e-06 0.000134483 2.06795e-06 4.11612e-05 2.4667e-05 7.26248e-06 2.37974e-05 0.000134513 0.00142482 0.00011707 0.000161848 0.000395011 3.01961e-05 3.95325e-05 3.12398e-06 3.57709e-05 1.36277e-06 1.01068e-05 0.000350805 5.09019e-05 0.000805241 6.60321e-05 2.13734e-05 9.88654e-05 2.1991e-06 3.24065e-05 3.9479e-05 4.45178e-05 0.00205024 0.000780899 2.0633e-05 1.89997e-05 0.00197261 0.000259391
  ```

## 模型加载

首先从文件系统中读取MindSpore Lite模型，并通过`mindspore::lite::Model::Impor`t函数导入模型进行解析。

```c++
// Read model file.
size_t size = 0;
char *model_buf = ReadFile(model_path, &size);
if (model_buf == nullptr) {
  std::cerr << "Read model file failed." << std::endl;
  return RET_ERROR;
}
// Load the .ms model.
auto model = mindspore::lite::Model::Import(model_buf, size);
delete[](model_buf);
if (model == nullptr) {
  std::cerr << "Import model file failed." << std::endl;
  return RET_ERROR;
}
```

## 模型编译

模型编译主要包括创建配置上下文、创建会话、图编译等步骤。

```c++
mindspore::session::LiteSession *Compile(mindspore::lite::Model *model) {
  // Create and init context.
  auto context = std::make_shared<mindspore::lite::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed while." << std::endl;
    return nullptr;
  }

  // Create the session.
  mindspore::session::LiteSession *session = mindspore::session::LiteSession::CreateSession(context.get());
  if (session == nullptr) {
    std::cerr << "CreateSession failed while running." << std::endl;
    return nullptr;
  }

  // Compile graph.
  auto ret = session->CompileGraph(model);
  if (ret != mindspore::lite::RET_OK) {
    delete session;
    std::cerr << "Compile failed while running." << std::endl;
    return nullptr;
  }

  // Note: when use model->Free(), the model can not be compiled again.
  if (model != nullptr) {
    model->Free();
  }
  return session;
}
```

## 模型推理

模型推理主要包括输入数据、执行推理、获得输出等步骤，其中本示例中的输入数据是通过随机数据构造生成，最后将执行推理后的输出结果打印出来。

```c++
int Run(mindspore::session::LiteSession *session) {
  auto inputs = session->GetInputs();
  auto ret = GenerateInputDataWithRandom(inputs);
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
    return ret;
  }

  ret = session->RunGraph();
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Inference error " << ret << std::endl;
    return ret;
  }

  auto out_tensors = session->GetOutputs();
  for (auto tensor : out_tensors) {
    std::cout << "tensor name is:" << tensor.first << " tensor size is:" << tensor.second->Size()
              << " tensor elements num is:" << tensor.second->ElementsNum() << std::endl;
    auto out_data = reinterpret_cast<float *>(tensor.second->MutableData());
    std::cout << "output data is:";
    for (int i = 0; i < tensor.second->ElementsNum() && i <= 50; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
  return mindspore::lite::RET_OK;
}
```

## 内存释放

无需使用MindSpore Lite推理框架时，需要释放已经创建的`LiteSession`和`Model`。

```c++
// Delete model buffer.
delete model;
// Delete session buffer.
delete session;
```