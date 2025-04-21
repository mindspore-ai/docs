# Experiencing C-language Simplified Inference Demo

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/infer/quick_start_c.md)

## Overview

This tutorial provides a sample program for MindSpore Lite to perform inference, which demonstrates the basic process of end-side inference with C-language by randomly typing, performing inference, and printing inference results, so that users can quickly understand the use of MindSpore Lite to perform inference-related APIs. This tutorial performs the inference of MobileNetV2 model by taking randomly generated data as input data and prints obtained output data. The related code is in the [mindspore/lite/examples/quick_start_c](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/quick_start_c) directory.

Performing inference with MindSpore Lite consists of the following main steps:

1. Read model: Read the `.ms` model converted by [Model Conversion Tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html) from the file system.
2. Create configuration context: Create a Configuration [Context](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_c/context_c.html) that holds some basic configuration parameters needed, to guide model compilation and model execution.
3. Create, load and compile Model: Before executing inference, you need to call [MSModelBuildFromFile](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_c/model_c.html#msmodelbuildfromfile) interface of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_c/model_c.html) for model loading and compilation, and configure the Context obtained in the previous step into the Model. The model loading phase parses the file cache into a runtime model. The model compilation phase mainly carries out the process of operator selection scheduling, subgraph slicing, etc, which will consume more time, so it is recommended that the Model be created once, compiled once, and reasoned several times.
4. Input data: The data needs to be padded in the `Input Tensor` before model execution.
5. Execute inference: Use [MSModelPredict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/function_model_c.h_MSModelPredict-1.html) inferene of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_c/model_c.html) for model inference.
6. Obtain output: After the model execution, the inference result can be obtained by `output Tensor`.
7. Free memory: When do not need to use MindSpore Lite inference framework, you need to free the created [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_c/model_c.html).

![img](../images/lite_runtime.png)

## Building and Running the Demo

### Linux X86

- Environment requirements

    - System environment: Linux x86_64, Ubuntu 18.04.02LTS recommended
    - Compilation dependencies:
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- Compiling and building

  Execute the [build script](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/quick_start_c/build.sh) in `mindspore/lite/examples/quick_start_c` directory, which will automatically download the MindSpore Lite inference framework library and the model file and compile the Demo.

  ```bash
  bash build.sh
  ```

  > If the build script fails to download the MindSpore Lite inference framework, please manually download the MindSpore Lite model inference framework [mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) for the CPU hardware platform and Ubuntu-x64 operating system, after decompression copy the `libmindspore-lite.so` file from the unpacked `runtime/lib` directory to `mindspore/ lite/examples/quick_start_c/lib` directory, and the files in `runtime/include` directory to `mindspore/lite/examples/quick_start_c/include` directory, and copy the `libmindspore_glog.so.0` file from the `runtime/third_party/glog` directory to the `libmindspore_glog.so` file in `mindspore/ lite/examples/quick_start_c/lib` directory.
  >
  > If the build script fails to download the MobileNetV2 model, please manually download the relevant model file [mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms) and copy it to the `mindspore/lite/examples/quick_start_c/model` directory.
  >
  > After manually downloading and placing the files in the specified location, the build.sh script needs to be executed again to complete the compiling and building.

- Executing inference

  After compiling and building, go to the `mindspore/lite/examples/quick_start_c/build` directory and execute the following command to experience the MobileNetV2 model inference by MindSpore Lite.

  ```bash
  ./mindspore_quick_start_c ../model/mobilenetv2.ms
  ```

  When the execution is completed, the following results will be obtained. Print the name, the size and the number of the output Tensor and the first 50 data:  

  ```text
  Tensor name: Softmax-65, tensor size is 4004 ,elements num: 1001.
  output data is:
  0.000011 0.000015 0.000015 0.000079 0.000070 0.000702 0.000120 0.000590 0.000009 0.000004 0.000004 0.000002 0.000002 0.000002 0.000010 0.000055 0.000006 0.000010 0.000003 0.000010 0.000002 0.000005 0.000001 0.000002 0.000004 0.000006 0.000008 0.000003 0.000015 0.000005 0.000011 0.000020 0.000006 0.000002 0.000011 0.000170 0.000005 0.000009 0.000006 0.000002 0.000003 0.000009 0.000005 0.000006 0.000003 0.000011 0.000005 0.000027 0.000003 0.000050 0.000016
  ```

### Windows

- Environment requirements

    - System environment: Windows 7, Windows 10; 64-bit
    - Compilation dependencies:
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [MinGW GCC](https://sourceforge.net/projects/mingw-w64/files/ToolchainstargettingWin64/PersonalBuilds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z/download) = 7.3.0

- Compiling and building

    - Library downloading: Please manually download the MindSpore Lite model inference framework [mindspore-lite-{version}-win-x64.zip](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) with CPU as the hardware platform and Windows-x64 as the operating system, after decompression copy all files in the `runtime\lib` directory to the `mindspore\lite\examples\quick_start_clib\` project directory, and the files in the `runtime\include` directory to the `mindspore\lite\examples\quick_start_c\include` project directory. (Note: the `lib` and `include` directories under the project need to be created manually)

    - Model downloading: Please manually download the relevant model file [mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms) and copy it to the `mindspore\ lite\examples\quick_start_c\model` directory.

    - Compiling: Execute the [build script](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/quick_start_c/build.bat) in the `mindspore\lite\examples\quick_start_c` directory, which can automatically download the relevant files and compile the Demo.

  ```bash
  call build.bat
  ```

- Executing inference

  After compiling and building, go to the `mindspore\lite\examples\quick_start_c\build` directory and execute the following command to experience the MobileNetV2 model inference by MindSpore Lite.

  ```bash
  set PATH=..\lib;%PATH%
  call mindspore_quick_start_c.exe ..\model\mobilenetv2.ms
  ```

  When the execution is completed, the following results will be obtained. Print the name, the size and the number of the output Tensor and the first 50 data:

  ```text
  Tensor name: Softmax-65, tensor size is 4004 ,elements num: 1001.
  output data is:
  0.000011 0.000015 0.000015 0.000079 0.000070 0.000702 0.000120 0.000590 0.000009 0.000004 0.000004 0.000002 0.000002 0.000002 0.000010 0.000055 0.000006 0.000010 0.000003 0.000010 0.000002 0.000005 0.000001 0.000002 0.000004 0.000006 0.000008 0.000003 0.000015 0.000005 0.000011 0.000020 0.000006 0.000002 0.000011 0.000170 0.000005 0.000009 0.000006 0.000002 0.000003 0.000009 0.000005 0.000006 0.000003 0.000011 0.000005 0.000027 0.000003 0.000050 0.000016
  ```

## Configuring CMake

The following is sample code when the `libmindspore-lite.so` static library is integrated via CMake.

> The `-wl,--whole-archive` option needs to be passed to the linker when the `libmindspore-lite.so` static library is integrated.
>
> Since the `-fstack-protector-strong` stack-protected compiling option was added when compiling MindSpore Lite, it is also necessary to link the `ssp` library in MinGW on the Windows platform.
>
> Since support for so library file handling was added when compiling MindSpore Lite, it is also necessary to link the `dl` library on the Linux platform.

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

## Creating Configuration Context

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

## Creating, Loading and Compiling Model

Model loading and compilation can be done by calling [MSModelBuildFromFile](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_c/model_c.html#msmodelbuildfromfile)  interface of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_c/model_c.html) to load and compile from the file path to get the runtime model. In this case, `argv[1]` corresponds to the model file path inputted from the console.

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

## Model Inference

The model inference mainly includes the steps of input data, inference execution, and obtaining output, where the input data in this example is generated by random data, and finally the output result after execution of inference is printed.

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

## Freeing memory

When do not need to use MindSpore Lite inference framework, you need to free the `Model` that has been created.

```c
// Delete model.
MSModelDestroy(&model);
```
