# "编译"一个MNIST分类模型

 `Linux` `IoT` `C++` `全流程` `模型编译` `模型代码生成` `模型部署` `推理应用` `初级` `中级` `高级`

<!-- TOC -->

- ["编译"一个MNIST分类模型](#编译一个MNIST分类模型)
    - [概述](#概述)
    - [生成代码](#生成代码)
    - [部署应用](#部署应用)
        - [编译依赖](#编译依赖)
        - [构建与运行](#构建与运行)
            - [快速使用](#快速使用)
            - [生成代码工程说明](#生成代码工程说明)
            - [代码编译](#代码编译)
            - [代码部署](#代码部署)
        - [编写推理代码示例](#编写推理代码示例)
    - [更多详情](#更多详情)
        - [Android平台编译部署](#android平台编译部署)
        - [Arm&nbsp;Cortex-M平台编译部署](#armcortex-m平台编译部署)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_zh_cn/quick_start/quick_start_codegen.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

本教程介绍如何使用MindSpore Lite代码生成工具Codegen，快速生成以及部署轻量化推理代码。converter、codegen等工具的获取可以参考MindSpore团队构建文档中提供的[编译输出](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/build.html#id4)。

其主要流程如下图流程图所示：

![img](../images/lite_codegen.png)"

1. 使用训练框架，如MindSpore等，得到训练好的模型；
2. 使用MindSpore Lite转换工具converter，将预训练模型转换为`*.ms`格式文件；
3. 使用Codegen代码生成工具，输入`*.ms`文件自动生成推理代码；
4. 通过交叉编译，生成支持不同平台的推理库文件。

我们推荐从MNIST分类模型推理代码入手，了解Codegen生成代码、编译构建、部署等流程，从而达到快速入门的效果。

本教程基于MindSpore团队提供的Codegen代码生成工具以及算子库，演示了编译一个模型并构建部署的流程。

## 生成代码

首先下载[MNIST分类网络](https://download.mindspore.cn/model_zoo/official/lite/mnist_lite/mnist.ms)。使用Codegen编译MNIST分类模型，生成对应的x86平台推理代码。生成代码的具体命令如下：

```bash
./codegen --codePath=. --modelPath=mnist.ms --moduleName=mnist --target=x86
```

关于Codegen的更多使用命令说明，可参见[Codegen工具的详细介绍](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/code_generator.html)。

## 部署应用

接下来介绍如何构建MindSpore Lite Codegen生成的模型推理代码工程，并在x86平台完成部署。

### 编译依赖

- [CMake](https://cmake.org/download/) >= 3.18.3
- [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

### 构建与运行

#### 使用脚本一键生成代码、执行

下载[MindSpore源码](https://gitee.com/mindspore/mindspore)，进入`mindspore/mindspore/lite/micro/examples/mnist`目录并执行脚本`mnist.sh`自动生成模型推理代码并编译工程目录。

```bash
bash mnist.sh
```

推理结果如下：

```text
input 0: mnist_input.bin
51, 52, 68, 78, 78, 88, 75, 87, 68, 61, 51, 56, 63, 55, 66, 61, 56, 71, 57, 73,
mnist inference success.
```

#### 生成代码工程说明

进入`mindspore/mindspore/lite/micro/example/mnist`目录中。

1. 算子静态库目录说明

    在编译此工程之前需要预先获取x86平台对应的算子库[codegen](https://www.mindspore.cn/tutorial/lite/zh-CN/master/use/downloads.html)，解压后得到operator_library，将其拷贝到当前目录下。

    以本教程为例，预置x86平台算子静态库的目录如下：

    ```text
    ├── operator_library    # 对应平台算子库目录
        ├── include         # 对应平台算子库头文件目录
        └── lib             # 对应平台算子库静态库目录
    ```

2. 生成代码工程目录说明

    当前目录下预置了MNIST分类网络生成的代码。

    ```text
    ├── mnist               # 生成代码的根目录
        ├── benchmark       # 生成代码的benchmark目录
        └── src             # 模型推理代码目录
    ```

#### 代码编译

1. 编译生成模型静态库

    组织模型生成的推理代码以及算子静态库，编译生成模型推理静态库。

    进入代码工程src目录下并新建build目录：

    ```bash
    cd mnist/src && mkdir build
    ```

    进入build目录：

    ```bash
    cd build
    ```

    开始编译：

    ```bash
    cmake -DOP_LIB={path to}/operator_library/lib/x86/liboplib.a  \
        -DOP_HEADER_PATH={path to}/operator_library/include/    \
        ..
    make
    ```

    > {path to}需要用户根据实际情况填写。

    代码工程编译成功结果：

    ```text
    [100%] Linking C static library libmnist.a
    unzip raw static library libmnist.a
    raw static library libmnist.a size:
    -rw-r--r-- 1 root root 356K Mar  4 16:48 libmnist.a
    generate specified static library libmnist.a
    new static library libmnist.a size:
    -rw-r--r-- 1 root root 735K Mar  4 16:48 libmnist.a
    ```

    此时在mnist/src/build目录下生成了libmnist.a，推理执行库。

2. 编译生成可执行文件

    组织模型推理静态库以及benchmark代码，编译生成二进制可执行文件文件，进入mnist/benchmark目录并新建build目录：

    ```bash
    cd mnist/benchmark && mkdir build
    ```

    进入build目录并编译：

    ```bash
    cd build
    cmake -DMODEL_LIB=../../src/build/libmnist.a  ..
    make
    ```

    代码工程编译成功结果：

    ```text
    [100%] Linking C executable benchmark
    [100%] Built target benchmark
    ```

    此时在mnist/benchmark/build目录下生成了benchmark可执行文件。

#### 代码部署

本示例部署于x86平台。由代码工程编译成功以后的产物为`benchmark`可执行文件，将其拷贝到用户的目标Linux服务器中即可执行。在目标Linux服务上执行编译成功的二进制文件:

```bash
./benchmark mnist_input.bin mnist.net
```

> mnist_input.bin在example/mnist目录下，mnist.net为模型参数文件，在example/mnist/src目录下。
生成结果如下：

```text
input 0: mnist_input.bin
51, 52, 68, 78, 78, 88, 75, 87, 68, 61, 51, 56, 63, 55, 66, 61, 56, 71, 57, 73,
mnist inference success.
```

### 编写推理代码示例

本教程中的`benchmark`内部实现主要用于指导用户如何编写以及调用Codegen"编译"的模型推理代码接口。

以下为接口调用的详细介绍，详情代码可以参见[examples/mnist](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/micro/example/mnist)下的示例代码示例：

#### 构建推理的上下文以及会话

本教程生成的代码为非并行代码，无需上下文context,可直接设为空。

```cpp
  size_t model_size = 0;
  Context *context = nullptr;
  session::LiteSession *session = mindspore::session::LiteSession::CreateSession(model_buffer, model_size, context);
  if (session == nullptr) {
      std::cerr << "create lite session failed" << std::endl;
      return RET_ERROR;
  }
```

#### 输入数据准备

用户所需要准备的输入数据内存空间，若输入是持久化文件，可通过读文件方式获取。若输入数据已经存在内存中，则此处无需读取，可直接传入数据指针。

```cpp
  std::vector<MSTensor *> inputs = session->GetInputs();
  MSTensor *input = inputs.at(0);
  if (input == nullptr) {
      return RET_ERROR;
  }
  // Assume we have got input data in memory.
  memcpy(input->MutableData(), input_buffer, input->Size());
```

#### 执行推理

```cpp
  session->RunGraph();
```

#### 推理结束获取输出

```cpp
  MSTensor *output = session->GetOutputs();
```

#### 释放内存session

```cpp
  delete session;
```

#### 推理代码整体调用流程

```cpp
  // Assume we have got model_buffer data in memory.
  size_t model_size = 0;
  Context *context = nullptr;
  session::LiteSession *session = mindspore::session::LiteSession::CreateSession(model_buffer, model_size, context);
  if (session == nullptr) {
      std::cerr << "create lite session failed" << std::endl;
      return RET_ERROR;
  }

  std::vector<MSTensor *> inputs = session->GetInputs();
  MSTensor *input = inputs.at(0);
  if (input == nullptr) {
      return RET_ERROR;
  }
  // Assume we have got input data in memory.
  memcpy(input->MutableData(), input_buffer, input->Size());

  session->RunGraph();

  auto outputs = session->GetOutputs();

  delete session;
```

## 更多详情

### [Android平台编译部署](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/micro/example/mobilenetv2/README.md)

### [Arm&nbsp;Cortex-M平台编译部署](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/micro/example/mnist/README.md)
