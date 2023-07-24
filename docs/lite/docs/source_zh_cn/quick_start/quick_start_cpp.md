# 体验C++极简推理Demo

`Linux` `Windows` `X86` `C++` `全流程` `推理应用` `数据准备` `初级`

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/docs/source_zh_cn/quick_start/quick_start_cpp.md)

## 概述

本教程提供了MindSpore Lite执行推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了C++进行端侧推理的基本流程，用户能够快速了解MindSpore Lite执行推理相关API的使用。本教程通过随机生成的数据作为输入数据，执行MobileNetV2模型的推理，打印获得输出数据。相关代码放置在[mindspore/lite/examples/quick_start_cpp](https://gitee.com/mindspore/mindspore/tree/r1.3/mindspore/lite/examples/quick_start_cpp)目录。

使用MindSpore Lite执行推理主要包括以下步骤：

1. 模型加载：从文件系统中读取由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/converter_tool.html)转换得到的`.ms`模型，通过[mindspore::lite::Model::Import](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/lite.html#import)导入模型，进行模型解析，创建得到 `Model *`。
2. 创建配置上下文：创建配置上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/lite.html#context)，保存会话所需的一些基本配置参数，用于指导图编译和图执行。
3. 创建会话：创建[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/session.html#litesession)会话，并将上一步得到的[Context](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/lite.html#context)配置到会话中。
4. 图编译：执行推理之前，需要调用[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/session.html#litesession)的`CompileGraph`接口进行图编译。图编译阶段主要进行子图切分、算子选型调度等过程，该阶段会耗费较多时间，所以建议[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/session.html#litesession)创建一次，编译一次，多次推理。
5. 输入数据：图执行之前需要向`输入Tensor`中填充数据。
6. 执行推理：使用[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/session.html#litesession)的`RunGraph`进行模型推理。
7. 获得输出：图执行结束之后，可以通过`输出Tensor`得到推理结果。
8. 释放内存：无需使用MindSpore Lite推理框架时，需要释放已创建的[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/session.html#litesession)和[Model](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/lite.html#model)。

![img](../images/lite_runtime.png)

> 如需查看MindSpore Lite高级用法，请参考[使用Runtime执行推理（C++）](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/runtime_cpp.html)。

## 构建与运行

### Linux X86

- 环境要求

    - 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- 编译构建

  在`mindspore/lite/examples/quick_start_cpp`目录下执行[build脚本](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/examples/quick_start_cpp/build.sh)，将自动下载MindSpore Lite推理框架库以及文模型文件并编译Demo。

  ```bash
  bash build.sh
  ```

  > 若使用该build脚本下载MindSpore Lite推理框架失败，请手动下载硬件平台为CPU、操作系统为Ubuntu-x64的MindSpore Lite 模型推理框架[mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/downloads.html)，将解压后`runtime/lib`目录下的`libmindspore-lite.so`文件拷贝到`mindspore/lite/examples/quick_start_cpp/lib`目录、`runtime/include`目录里的文件拷贝到`mindspore/lite/examples/quick_start_cpp/include`目录下。
  >
  > 若MobileNetV2模型下载失败，请手动下载相关模型文件[mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms)，并将其拷贝到`mindspore/lite/examples/quick_start_cpp/model`目录。
  >
  > 通过手动下载并且将文件放到指定位置后，需要再次执行build.sh脚本才能完成编译构建。

- 执行推理

  编译构建后，进入`mindspore/lite/examples/quick_start_cpp/build`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  ./mindspore_quick_start_cpp ../model/mobilenetv2.ms
  ```

  执行完成后将能得到如下结果，打印输出Tensor的名称、输出Tensor的大小，输出Tensor的数量以及前50个数据：

  ```text
  tensor name is:Default/head-MobileNetV2Head/Softmax-op204 tensor size is:4000 tensor elements num is:1000
  output data is:5.26823e-05 0.00049752 0.000296722 0.000377607 0.000177048 8.02107e-05 0.000212864 0.000422286 0.000273189 0.000234105 0.00099807 0.0042331 0.00204993 0.00124968 0.00294458 0.00139795 0.00111545 0.000656357 0.000809457 0.00153731 0.000621049 0.00224637 0.00127045 0.00187557 0.000420144 0.000150638 0.000266477 0.000438628 0.000187773 0.00054668 0.000212853 0.000921661 0.000127179 0.000565873 0.00100394 0.000300159 0.000282677 0.000358067 0.00215288 0.000477845 0.00107596 0.00065134 0.000722132 0.000807501 0.000631415 0.00043247 0.00125898 0.000255094 8.2606e-05 9.91917e-05 0.000794512
  ```

### Windows

- 环境要求

    - 系统环境：Windows 7，Windows 10；64位。
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [MinGW GCC](https://sourceforge.net/projects/mingw-w64/files/ToolchainstargettingWin64/PersonalBuilds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z/download) = 7.3.0

- 编译构建

    - 库下载：请手动下载硬件平台为CPU、操作系统为Windows-x64的MindSpore Lite模型推理框架[mindspore-lite-{version}-win-x64.zip](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/downloads.html)，将解压后`runtime/lib`目录下的所有文件拷贝到`mindspore/lite/examples/quick_start_cpp/lib`工程目录、`runtime/include`目录里的文件拷贝到`mindspore/lite/examples/quick_start_cpp/include`工程目录下。（注意：工程项目下的`lib`、`include`目录需手工创建）
    - 模型下载：请手动下载相关模型文件[mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms)，并将其拷贝到`mindspore/lite/examples/quick_start_cpp/model`目录。

    - 编译：在`mindspore/lite/examples/quick_start_cpp`目录下执行[build脚本](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/examples/quick_start_cpp/build.bat)，将能够自动下载相关文件并编译Demo。

  ```bash
  call build.bat
  ```

- 执行推理

  编译构建后，进入`mindspore/lite/examples/quick_start_cpp/build`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  set PATH=../lib;%PATH%
  call mindspore_quick_start_cpp.exe ../model/mobilenetv2.ms
  ```

  执行完成后将能得到如下结果，打印输出Tensor的名称、输出Tensor的大小，输出Tensor的数量以及前50个数据：

  ```text
  tensor name is:Default/head-MobileNetV2Head/Softmax-op204 tensor size is:4000 tensor elements num is:1000
  output data is:5.26823e-05 0.00049752 0.000296722 0.000377607 0.000177048 8.02107e-05 0.000212864 0.000422286 0.000273189 0.000234105 0.00099807 0.0042331 0.00204993 0.00124968 0.00294458 0.00139795 0.00111545 0.000656357 0.000809457 0.00153731 0.000621049 0.00224637 0.00127045 0.00187557 0.000420144 0.000150638 0.000266477 0.000438628 0.000187773 0.00054668 0.000212853 0.000921661 0.000127179 0.000565873 0.00100394 0.000300159 0.000282677 0.000358067 0.00215288 0.000477845 0.00107596 0.00065134 0.000722132 0.000807501 0.000631415 0.00043247 0.00125898 0.000255094 8.2606e-05 9.91917e-05 0.000794512
  ```

## 配置CMake

以下是通过CMake集成`libmindspore-lite.a`静态库时的示例代码。

> 集成`libmindspore-lite.a`静态库时需要将`-Wl,--whole-archive`的选项传递给链接器。
>
> 由于在编译MindSpore Lite的时候增加了`-fstack-protector-strong`栈保护的编译选项，所以在Windows平台上还需要链接MinGW中的`ssp`库。
>
> 由于在编译MindSpore Lite的时候增加了对so库文件处理的支持，所以在Linux平台上还需要链接`dl`库。

```cmake
cmake_minimum_required(VERSION 3.18.3)
project(QuickStartCpp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

# Add directory to include search path
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add directory to link search path
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

file(GLOB_RECURSE QUICK_START_CXX ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_executable(mindspore_quick_start_cpp ${QUICK_START_CXX})

target_link_libraries(
        mindspore_quick_start_cpp
        -Wl,--whole-archive mindspore-lite -Wl,--no-whole-archive
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

## 模型加载

模型加载需要从文件系统中读取MindSpore Lite模型，并通过`mindspore::lite::Model::Import`函数导入模型进行解析。

```c++
// Read model file.
size_t size = 0;
char *model_buf = ReadFile(model_path, &size);
if (model_buf == nullptr) {
  std::cerr << "Read model file failed." << std::endl;
  return -1;
}
// Load the .ms model.
auto model = mindspore::lite::Model::Import(model_buf, size);
delete[](model_buf);
if (model == nullptr) {
  std::cerr << "Import model file failed." << std::endl;
  return -1;
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

  // Generate random data as input data.
  auto ret = GenerateInputDataWithRandom(inputs);
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
    return ret;
  }

  // Run Inference.
  ret = session->RunGraph();
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Inference error " << ret << std::endl;
    return ret;
  }

  // Get Output Tensor Data.
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
