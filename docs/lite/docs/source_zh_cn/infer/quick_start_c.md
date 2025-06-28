# 体验C语言接口极简推理Demo

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/infer/quick_start_c.md)

## 概述

本教程提供了MindSpore Lite执行推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了C语言进行端侧推理的基本流程，用户能够快速了解MindSpore Lite执行推理相关API的使用。本教程通过随机生成的数据作为输入数据，执行MobileNetV2模型的推理，打印获得输出数据。相关代码放置在[mindspore/lite/examples/quick_start_c](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_c)目录。

使用MindSpore Lite执行推理主要包括以下步骤：

1. 模型读取：从文件系统中读取由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/converter/converter_tool.html)转换得到的`.ms`模型。
2. 创建配置上下文：创建配置上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_c/context_c.html)，保存需要的一些基本配置参数，用于指导模型编译和模型执行。
3. 模型创建、加载与编译：执行推理之前，需要调用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_c/model_c.html)的[MSModelBuildFromFile](https://www.mindspore.cn/lite/api/zh-CN/master/api_c/model_c.html#msmodelbuildfromfile)接口进行模型加载和模型编译，并将上一步得到的Context配置到Model中。模型加载阶段将文件缓存解析成运行时的模型。模型编译阶段主要进行算子选型调度、子图切分等过程，该阶段会耗费较多时间，所以建议Model创建一次，编译一次，多次推理。
4. 输入数据：模型执行之前需要向`输入Tensor`中填充数据。
5. 执行推理：使用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_c/model_c.html)的[MSModelPredict](https://www.mindspore.cn/lite/api/zh-CN/master/api_c/model_c.html#msmodelpredict)接口进行模型推理。
6. 获得输出：模型执行结束之后，可以通过`输出Tensor`得到推理结果。
7. 释放内存：无需使用MindSpore Lite推理框架时，需要释放已创建的[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_c/model_c.html)。

![img](../images/lite_runtime.png)

## 构建与运行

### Linux X86

- 环境要求

    - 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- 编译构建

  在`mindspore/lite/examples/quick_start_c`目录下执行[build脚本](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_c/build.sh)，将自动下载MindSpore Lite推理框架库以及模型文件并编译Demo。

  ```bash
  bash build.sh
  ```

  > 若使用该build脚本下载MindSpore Lite推理框架失败，请手动下载硬件平台为CPU、操作系统为Ubuntu-x64的MindSpore Lite 模型推理框架[mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)，将解压后`runtime/lib`目录下的`libmindspore-lite.so`文件拷贝到`mindspore/lite/examples/quick_start_c/lib`目录、`runtime/include`目录里的文件拷贝到`mindspore/lite/examples/quick_start_c/include`目录下、`runtime/third_party/glog/`目录下的`libmindspore_glog.so.0`文件拷贝到`mindspore/lite/examples/quick_start_c/lib`目录下的`libmindspore_glog.so`。
  >
  > 若MobileNetV2模型下载失败，请手动下载相关模型文件[mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms)，并将其拷贝到`mindspore/lite/examples/quick_start_c/model`目录。
  >
  > 通过手动下载并且将文件放到指定位置后，需要再次执行build.sh脚本才能完成编译构建。

- 执行推理

  编译构建后，进入`mindspore/lite/examples/quick_start_c/build`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  ./mindspore_quick_start_c ../model/mobilenetv2.ms
  ```

  执行完成后将能得到如下结果，打印输出Tensor的名称、输出Tensor的大小，输出Tensor的数量以及前50个数据：

  ```text
  Tensor name: Softmax-65, tensor size is 4004, elements num: 1001.
  output data is:
  0.000011 0.000015 0.000015 0.000079 0.000070 0.000702 0.000120 0.000590 0.000009 0.000004 0.000004 0.000002 0.000002 0.000002 0.000010 0.000055 0.000006 0.000010 0.000003 0.000010 0.000002 0.000005 0.000001 0.000002 0.000004 0.000006 0.000008 0.000003 0.000015 0.000005 0.000011 0.000020 0.000006 0.000002 0.000011 0.000170 0.000005 0.000009 0.000006 0.000002 0.000003 0.000009 0.000005 0.000006 0.000003 0.000011 0.000005 0.000027 0.000003 0.000050 0.000016
  ```

### Windows

- 环境要求

    - 系统环境：Windows 7，Windows 10；64位。
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [MinGW GCC](https://sourceforge.net/projects/mingw-w64/files/ToolchainstargettingWin64/PersonalBuilds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z/download) = 7.3.0

- 编译构建

    - 库下载：请手动下载硬件平台为CPU、操作系统为Windows-x64的MindSpore Lite模型推理框架[mindspore-lite-{version}-win-x64.zip](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)，将解压后`runtime\lib`目录下的所有文件拷贝到`mindspore\lite\examples\quick_start_c\lib`工程目录、`runtime\include`目录里的文件拷贝到`mindspore\lite\examples\quick_start_c\include`工程目录下。（注意：工程项目下的`lib`、`include`目录需手工创建）

    - 模型下载：请手动下载相关模型文件[mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms)，并将其拷贝到`mindspore\lite\examples\quick_start_c\model`目录。

    - 编译：在`mindspore\lite\examples\quick_start_c`目录下执行[build脚本](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_c/build.bat)，将能够自动下载相关文件并编译Demo。

  ```bash
  call build.bat
  ```

- 执行推理

  编译构建后，进入`mindspore\lite\examples\quick_start_c\build`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  set PATH=..\lib;%PATH%
  call mindspore_quick_start_c.exe ..\model\mobilenetv2.ms
  ```

  执行完成后将能得到如下结果，打印输出Tensor的名称、输出Tensor的大小，输出Tensor的数量以及前50个数据：

  ```text
  Tensor name: Softmax-65, tensor size is 4004, elements num: 1001.
  output data is:
  0.000011 0.000015 0.000015 0.000079 0.000070 0.000702 0.000120 0.000590 0.000009 0.000004 0.000004 0.000002 0.000002 0.000002 0.000010 0.000055 0.000006 0.000010 0.000003 0.000010 0.000002 0.000005 0.000001 0.000002 0.000004 0.000006 0.000008 0.000003 0.000015 0.000005 0.000011 0.000020 0.000006 0.000002 0.000011 0.000170 0.000005 0.000009 0.000006 0.000002 0.000003 0.000009 0.000005 0.000006 0.000003 0.000011 0.000005 0.000027 0.000003 0.000050 0.000016
  ```

## 配置CMake

以下是通过CMake集成`libmindspore-lite.so`静态库时的示例代码。

> 集成`libmindspore-lite.so`静态库时需要将`-Wl,--whole-archive`的选项传递给链接器。
>
> 由于在编译MindSpore Lite的时候增加了`-fstack-protector-strong`栈保护的编译选项，所以在Windows平台上还需要链接MinGW中的`ssp`库。
>
> 由于在编译MindSpore Lite的时候增加了对so库文件处理的支持，所以在Linux平台上还需要链接`dl`库。

```cmake
cmake_minimum_required(VERSION 3.18.3)
project(QuickStartC)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3.0)
    message(FATAL_ERROR "GCC version ${CMAKE_CXX_COMPILER_VERSION} must not be less than 7.3.0")
endif()

# Add directory to include search path
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add directory to linker search path
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

file(GLOB_RECURSE QUICK_START_CXX ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
add_executable(mindspore_quick_start_c ${QUICK_START_CXX})

target_link_libraries(
        mindspore_quick_start_c
        -Wl,--whole-archive libmindspore-lite -Wl,--no-whole-archive
        pthread
)

# Due to the increased compilation options for stack protection,
# it is necessary to target link ssp library when Use the static library in Windows.
if(WIN32)
    target_link_libraries(
            mindspore_quick_start_c
            ssp
    )
else()
    target_link_libraries(
            mindspore_quick_start_c
            dl
    )
endif()
```

## 创建配置上下文

```c
  // Create and init context, add CPU device info
  MSContextHandle context = MSContextCreate();
  if (context == NULL) {
    printf("MSContextCreate failed.\n");
    return kMSStatusLiteError;
  }
  const int thread_num = 2;
  MSContextSetThreadNum(context, thread_num);
  MSContextSetThreadAffinityMode(context, 1);

  MSDeviceInfoHandle cpu_device_info = MSDeviceInfoCreate(kMSDeviceTypeCPU);
  if (cpu_device_info == NULL) {
    printf("MSDeviceInfoCreate failed.\n");
    MSContextDestroy(&context);
    return kMSStatusLiteError;
  }
  MSDeviceInfoSetEnableFP16(cpu_device_info, false);
  MSContextAddDeviceInfo(context, cpu_device_info);
```

## 模型创建加载与编译

模型加载与编译可以调用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_c/model_c.html)的[MSModelBuildFromFile](https://www.mindspore.cn/lite/api/zh-CN/master/api_c/model_c.html#msmodelbuildfromfile)接口，从文件路径加载、编译得到运行时的模型。本例中`argv[1]`对应的是从控制台中输入的模型文件路径。

```c
  // Create model
  MSModelHandle model = MSModelCreate();
  if (model == NULL) {
    printf("MSModelCreate failed.\n");
    MSContextDestroy(&context);
    return kMSStatusLiteError;
  }

  // Build model
  int ret = MSModelBuildFromFile(model, argv[1], kMSModelTypeMindIR, context);
  if (ret != kMSStatusSuccess) {
    printf("MSModelBuildFromFile failed, ret: %d.\n", ret);
    MSModelDestroy(&model);
    return ret;
  }
```

## 模型推理

模型推理主要包括输入数据、执行推理、获得输出等步骤，其中本示例中的输入数据是通过随机数据构造生成，最后将执行推理后的输出结果打印出来。

```c
  // Get Inputs
  MSTensorHandleArray inputs = MSModelGetInputs(model);
  if (inputs.handle_list == NULL) {
    printf("MSModelGetInputs failed, ret: %d.\n", ret);
    MSModelDestroy(&model);
    return ret;
  }

  // Generate random data as input data.
  ret = GenerateInputDataWithRandom(inputs);
  if (ret != kMSStatusSuccess) {
    printf("GenerateInputDataWithRandom failed, ret: %d.\n", ret);
    MSModelDestroy(&model);
    return ret;
  }

  // Model Predict
  MSTensorHandleArray outputs;
  ret = MSModelPredict(model, inputs, &outputs, NULL, NULL);
  if (ret != kMSStatusSuccess) {
    printf("MSModelPredict failed, ret: %d.\n", ret);
    MSModelDestroy(&model);
    return ret;
  }

  // Print Output Tensor Data.
  for (size_t i = 0; i < outputs.handle_num; ++i) {
    MSTensorHandle tensor = outputs.handle_list[i];
    int64_t element_num = MSTensorGetElementNum(tensor);
    printf("Tensor name: %s, tensor size is %ld ,elements num: %ld.\n", MSTensorGetName(tensor),
           MSTensorGetDataSize(tensor), element_num);
    const float *data = (const float *)MSTensorGetData(tensor);
    printf("output data is:\n");
    const int max_print_num = 50;
    for (int j = 0; j < element_num && j <= max_print_num; ++j) {
      printf("%f ", data[j]);
    }
    printf("\n");
  }
```

## 内存释放

无需使用MindSpore Lite推理框架时，需要释放已经创建的`Model`。

```c
// Delete model.
MSModelDestroy(&model);
```
