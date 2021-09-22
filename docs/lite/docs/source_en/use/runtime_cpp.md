# Using C++ Interface to Perform Inference

`Windows` `macOS` `Linux` `iOS` `Android` `C++` `Inference Application` `Model Loading` `Data Preparation` `Intermediate` `Expert`

<!-- TOC -->

- [Using C++ Interface to Perform Inference](#using-c-interface-to-perform-inference)
    - [Overview](#overview)
    - [Model Reading](#model-reading)
    - [Creating and Configuring Context](#creating-and-configuring-context)
        - [Configuring the Number of Threads](#configuring-the-number-of-threads)
        - [Configuring the Thread Affinity](#configuring-the-thread-affinity)
        - [Configuring the Parallelization](#configuring-the-parallelization)
        - [Configuring the GPU Backend](#configuring-the-gpu-backend)
        - [Configuring the NPU Backend](#configuring-the-npu-backend)
        - [Configuring the TensorRT Backend](#configuring-the-tensorrt-backend)
    - [Model Creating Loading and Building](#model-creating-loading-and-building)
    - [Inputting Data](#inputting-data)
    - [Executing Inference](#executing-inference)
    - [Obtaining Output](#obtaining-output)
    - [Releasing Memory](#releasing-memory)
    - [Advanced Usage](#advanced-usage)
        - [Resizing the Input Dimension](#resizing-the-input-dimension)
        - [Parallel Models](#parallel-models)
        - [Sharing a Memory Pool](#sharing-a-memory-pool)
        - [Calling Back a Model During the Running Process](#calling-back-a-model-during-the-running-process)
        - [Separating Graph Loading and Model Build](#separating-graph-loading-and-model-build)
        - [Viewing Logs](#viewing-logs)
        - [Obtaining the Version Number](#obtaining-the-version-number)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/runtime_cpp.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

> MindSpore has unified the inference API. If you want to continue to use the MindSpore Lite independent API for inference, you can refer to the [document](https://www.mindspore.cn/lite/docs/en/r1.3/use/runtime_cpp.html).

## Overview

After the model is converted into a `.ms` model by using the MindSpore Lite model conversion tool, the inference process can be performed in Runtime. For details, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html). This tutorial describes how to use the [C++ API](https://www.mindspore.cn/lite/api/en/master/index.html) to perform inference.

To use the MindSpore Lite inference framework, perform the following steps:

1. Read the model: Read the `.ms` model file converted by the [model conversion tool](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html) from the file system.
2. Create and configure context: Create and configure [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) to save some basic configuration parameters required to build and execute the model.
3. Create, load and build a model: Use [Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build) of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to create and build the model, and configure the [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) obtained in the previous step. In the model loading phase, the file cache is parsed into a runtime model. In the model building phase, subgraph partition, operator selection and scheduling are performed, which will take a long time. Therefore, it is recommended that the model should be created once, built once, and performed for multiple times.
4. Input data: Before the model is executed, data needs to be filled in the `Input Tensor`.
5. Perform inference: Use [Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict) of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to perform model inference.
6. Obtain the output: After the model execution is complete, you can obtain the inference result by `Output Tensor`.
7. Release the memory: If the MindSpore Lite inference framework is not required, release the created [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model).

![img](../images/lite_runtime.png)

> For details about the calling process of MindSpore Lite inference, see [Simplified MindSpore Lite C++ Demo](https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start_cpp.html).

## Model Reading

When MindSpore Lite is used for model inference, read the `.ms` model file converted by using the model conversion tool from the file system and store it in the memory buffer. For details, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html).

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L332) demonstrates how to load a MindSpore Lite model from the file system:

```cpp
// Read model file.
size_t size = 0;
char *model_buf = ReadFile(model_path, &size);
if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
}
```

## Creating and Configuring Context

The context saves some basic configuration parameters required to build and execute the model. If you use `new` to create a [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) and do not need it any more, use `delete` to release it. Generally, the [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) is released after the [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) is created and built.

The default backend of MindSpore Lite is CPU. After Context is created, call [MutableDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mutabledeviceinfo) to return list of backend device information. Add the default [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_CPUDeviceInfo.html#class-cpudeviceinfo) to the list.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L250) demonstrates how to create a context, configure the default CPU backend, and enable CPU float16 inference.

```cpp
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
}
auto &device_list = context->MutableDeviceInfo();
auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
if (cpu_device_info == nullptr) {
  std::cerr << "New CPUDeviceInfo failed." << std::endl;
}
// CPU use float16 operator as priority.
cpu_device_info->SetEnableFP16(true);
device_list.push_back(cpu_device_info);
```

> `MutableDeviceInfo` supports multiple DeviceInfos, including [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_CPUDeviceInfo.html#class-cpudeviceinfo), [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_GPUDeviceInfo.html#class-gpudeviceinfo), [KirinNPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_KirinNPUDeviceInfo.html#class-kirinnpudeviceinfo). The device number limit is 3. During the inference, the operator will choose device in order.
>
> Float16 takes effect only when the CPU is under the ARM v8.2 architecture. Other models and x86 platforms that do not supported Float16 will be automatically rolled back to Float32.
>
> For the iOS platform, only the CPU backend is supported, and Float16 is temporarily not supported.

The advanced interfaces contained in [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) are defined as follows:

### Configuring the Number of Threads

Use [SetThreadNum](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setthreadnum) of [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) to configure the number of threads:

```cpp
// Configure the number of worker threads in the thread pool to 2, including the main thread.
context->SetThreadNum(2);
```

### Configuring the Thread Affinity

Use [SetThreadAffinity](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setthreadaffinity) of [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) to configure the thread affinity. If the parameter is `int mode`, configure the binding strategy. The effective value is 0-2, 0 means no core binding by default, 1 means preferential binding to large cores, and 2 means preferential binding to small cores. If the parameter is `const std::vector<int> &core_list`, configure the binding core list. When configuring at the same time, the core_list is effective, but the mode is not effective.

```cpp
// Configure the thread to be bound to the big core first.
// Valid value: 0: no affinities, 1: big cores first, 2: little cores first
context->SetThreadAffinity(1);
```

### Configuring the Parallelization

Use [SetEnableParallel](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setenableparallel) of [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) to configure whether to support parallelism when executing inference:

```cpp
// Configure the inference supports parallel.
context->SetEnableParallel(true);
```

### Configuring the GPU Backend

If the backend to be executed is GPUs, you need to set [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_GPUDeviceInfo.html#class-gpudeviceinfo) as the first choice. It is suggested to set [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_CPUDeviceInfo.html#class-cpudeviceinfo) as the second choice, to ensure model inference. Use `SetEnableFP16` to enable GPU Float16 inference.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L114) demonstrates how to create the CPU and GPU heterogeneous inference backend and how to enable Float16 inference for the GPU.

```cpp
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
}
auto &device_list = context->MutableDeviceInfo();

// Set GPU device first, make GPU preferred backend.
auto gpu_device_info = std::make_shared<mindspore::GPUDeviceInfo>();
if (gpu_device_info == nullptr) {
  std::cerr << "New GPUDeviceInfo failed." << std::endl;
}
// GPU use float16 operator as priority.
gpu_device_info->SetEnableFP16(true);
// Set VNIDIA device id, only valid when GPU backend is TensorRT.
gpu_device_info->SetDeviceID(0);
// The GPU device context needs to be push_back into device_list to work.
device_list.push_back(gpu_device_info);

// Set CPU device after GPU as second choice.
auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
if (cpu_device_info == nullptr) {
  std::cerr << "New CPUDeviceInfo failed." << std::endl;
}
// CPU use float16 operator as priority.
cpu_device_info->SetEnableFP16(true);
device_list.push_back(cpu_device_info);
```

> The current GPU backend distinguishes `arm64`and `x86_64`platforms.
>
> - On `arm64`, the backend of GPU is based on OpenCL. GPUs of Mali and Adreno are supported. The OpenCL version is 2.0.
>
> The configuration is as follows:
>
> CL_TARGET_OPENCL_VERSION=200
>
> CL_HPP_TARGET_OPENCL_VERSION=120
>
> CL_HPP_MINIMUM_OPENCL_VERSION=120
>
> - On `x86_64`, the backend of GPU is based on TensorRT. The TensorRT version is 6.0.1.5.
>
> Whether the attribute `SetEnableFP16` can be set successfully depends on the [CUDA computer capability](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix) of the current device.
>
> The attribute `SetDeviceID` only valid for TensorRT, used to specify the NVIDIA device ID.

### Configuring the NPU Backend

If the backend to be executed is NPUs, you need to set [KirinNPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_KirinNPUDeviceInfo.html#class-kirinnpudeviceinfo) as the first choice. It is suggested to set [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_CPUDeviceInfo.html#class-cpudeviceinfo) as the second choice, to ensure model inference. Use `SetFrequency` to set npu frequency.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L127) shows how to create the CPU and NPU heterogeneous inference backend and set the NPU frequency to 3. It can be set to 1 (low power consumption), 2 (balanced), 3 (high performance), 4 (extreme performance).

```cpp
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
}
auto &device_list = context->MutableDeviceInfo();

// Set NPU device first, make NPU preferred backend.
auto npu_device_info = std::make_shared<mindspore::KirinNPUDeviceInfo>();
if (npu_device_info == nullptr) {
  std::cerr << "New KirinNPUDeviceInfo failed." << std::endl;
}
// NPU set frequency to be 3.
npu_device_info->SetFrequency(3);
// The NPU device context needs to be push_back into device_list to work.
device_list.push_back(npu_device_info);

// Set CPU device after NPU as second choice.
auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
if (cpu_device_info == nullptr) {
  std::cerr << "New CPUDeviceInfo failed." << std::endl;
}
// CPU use float16 operator as priority.
cpu_device_info->SetEnableFP16(true);
device_list.push_back(cpu_device_info);
```

### Configuring the NNIE Backend

When the backend that needs to be executed is the heterogeneous inference based on CPU and NNIE, you only need to create the Context according to the configuration method of [CPU Backend](#creating-and-configuring-context) without specifying a provider.

## Model Creating Loading and Building

When MindSpore Lite is used for inference, [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) is the main entry for inference. You can use [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to load, build and execute model. Use the [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) created in the previous step to call the [Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build) of Model to load and build the runtime model.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L265) demonstrates how to create, load and build a model:

```cpp
// Create model
auto model = new (std::nothrow) mindspore::Model();
if (model == nullptr) {
  std::cerr << "New Model failed." << std::endl;
}
// Build model
auto build_ret = model->Build(model_buf, size, mindspore::kMindIR, context);
delete[](model_buf);
// After the model is built, the Context can be released.
...
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model failed." << std::endl;
}
```

> After the [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) is loaded and built, the [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) created in the previous step can be released.

## Inputting Data

Before executing a model, obtain the input [MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor) of the model and copy the input data to the input Tensor using `memcpy`. In addition, you can use the [DataSize](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#datasize) method to obtain the size of the data to be filled in to the tensor, use the [DataType](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#datatype) method to obtain the data type of the tensor, and use the [MutableData](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mutabledata) method to obtain the writable data pointer.

MindSpore Lite provides two methods to obtain the input tensor of a model.

1. Use the [GetInputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputbytensorname) method to obtain the input tensor based on the name. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L154) demonstrates how to call `GetInputByTensorName` to obtain the input tensor and fill in data.

   ```cpp
   // Pre-processing of input data, convert input data format to NHWC.
   ...
   // Assume that the model has only one input tensor named graph_input-173.
   auto in_tensor = model->GetInputByTensorName("graph_input-173");
   if (in_tensor.impl() == nullptr) {
       std::cerr << "Input tensor is nullptr" << std::endl;
   }
   auto input_data = in_tensor.MutableData();
   if (input_data == nullptr) {
       std::cerr << "MallocData for inTensor failed." << std::endl;
   }
   memcpy(in_data, input_buf, data_size);
   // Users need to free input_buf.
   ```

2. Use the [GetInputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputs) method to directly obtain the vectors of all model input tensors. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L137) demonstrates how to call `GetInputs` to obtain the input tensor and fill in data.

   ```cpp
   // Pre-processing of input data, convert input data format to NHWC.
   ...
   // Assume we have created a Model instance named model.
   auto inputs = model->GetInputs();
   // Assume that the model has only one input tensor.
   auto in_tensor = inputs.front();
   if (in_tensor == nullptr) {
       std::cerr << "Input tensor is nullptr" << std::endl;
   }
   auto *in_data = in_tensor.MutableData();
   if (in_data == nullptr) {
       std::cerr << "Data of in_tensor is nullptr" << std::endl;
   }
   memcpy(in_data, input_buf, data_size);
   // Users need to free input_buf.
   ```

> The data layout in the input tensor of the MindSpore Lite model must be `NHWC`. For more information about data pre-processing, see step 2 in [Writing On-Device Inference Code](https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start.html#writing-on-device-inference-code) in Android Application Development Based on JNI Interface to convert the input image into the Tensor format of the MindSpore model.
>
> [GetInputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputs) and [GetInputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputbytensorname) methods return data that do not need to be released by users.

## Executing Inference

Call the [Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict) function of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) for model inference.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L355) demonstrates how to call `Predict` to perform inference.

```cpp
auto inputs = model->GetInputs();
auto outputs = model->GetOutputs();
auto predict_ret = model->Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  std::cerr << "Predict error " << predict_ret << std::endl;
}
```

## Obtaining Output

After performing inference, MindSpore Lite can obtain the inference result of the model. MindSpore Lite provides three methods to obtain the output [MSTensor](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore.html#mstensor) of a model.

1. Use the [GetOutputsByNodeName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputsbynodename) method to obtain the vector of the tensor connected to the model output tensor based on the name of the model output node. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L170) demonstrates how to call `GetOutputsByNodeName` to obtain the output tensor.

   ```cpp
   // Assume we have created a Model instance named model before.
   // Assume that model has a output node named Softmax-65.
   auto output_vec = model->GetOutputsByNodeName("Softmax-65");
   // Assume that output node named Default/Sigmoid-op204 has only one output tensor.
   auto out_tensor = output_vec.front();
   if (out_tensor == nullptr) {
       std::cerr << "Output tensor is nullptr" << std::endl;
   }
   // Post-processing your result data.
   ```

2. Use the [GetOutputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputbytensorname) method to obtain the corresponding model output tensor based on the name of the model output tensor. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L200) demonstrates how to call `GetOutputsByTensorName` to obtain the output tensor.

   ```cpp
   // Assume we have created a Model instance named model.
   // We can use GetOutputTensorNames method to get all name of output tensor of model which is in order.
   auto tensor_names = model->GetOutputTensorNames();
   // Assume we have created a Model instance named model before.
   for (auto tensor_name : tensor_names) {
       auto out_tensor = model->GetOutputByTensorName(tensor_name);
       if (out_tensor == nullptr) {
           std::cerr << "Output tensor is nullptr" << std::endl;
       }
       // Post-processing the result data.
   }
   ```

3. Use the [GetOutputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputs) method to directly obtain the names of all model output tensors vector. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L226) demonstrates how to call `GetOutputs` to obtain the output tensor.

   ```cpp
   // Assume we have created a Model instance named model.
   auto out_tensors = model->GetOutputs();
   for (auto out_tensor : out_tensors) {
       // Post-processing the result data.
   }
   ```

> The data returned by the [GetOutputsByNodeName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputsbynodename), [GetOutputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputbytensorname), and [GetOutputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputs) methods does not need to be released by the user.

## Releasing Memory

If the MindSpore Lite inference framework is not required, you need to release the created Model. The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L370) demonstrates how to release the memory before the program ends.

```cpp
// Delete model.
// Assume that the variable of Model * is named model.
delete model;
```

## Advanced Usage

### Resizing the Input Dimension

When MindSpore Lite is used for inference, if the input shape needs to be resized, you can call the [Resize](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#resize) API of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to resize the shape of the input tensor after a model is created and built.

> Some networks do not support variable dimensions. As a result, an error message is displayed and the model exits unexpectedly. For example, the model contains the MatMul operator, one input tensor of the MatMul operator is the weight, and the other input tensor is the input. If a variable dimension API is called, the input tensor does not match the shape of the weight tensor. As a result, the inference fails.
>
> When the GPU backend is TensorRT, Resize only valid at dims NHW for NHWC format inputs, resize shape value should not be larger than the model inputs.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L321) demonstrates how to perform Resize on the input tensor of MindSpore Lite:

```cpp
// Assume we have created a Model instance named model.
auto inputs = model->GetInputs();
std::vector<int64_t> resize_shape = {1, 128, 128, 3};
// Assume the model has only one input,resize input shape to [1, 128, 128, 3]
std::vector<std::vector<int64_t>> new_shapes;
new_shapes.push_back(resize_shape);
return model->Resize(inputs, new_shapes);
```

### Parallel Models

MindSpore Lite supports parallel inference for multiple [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model). The thread pool and memory pool of each Mode are independent. However, multiple threads cannot call the [Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict) API of a single Model at the same time.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L470) demonstrates how to infer multiple Model in parallel:

```cpp
int RunModelParallel(const char *model_path) {
  size_t size = 0;
  char *model_buf = ReadFile(model_path, &size);
  if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
    return -1;
  }

  // Create and Build MindSpore model.
  auto model1 = CreateAndBuildModel(model_buf, size);
  auto model2 = CreateAndBuildModel(model_buf, size);
  delete[](model_buf);
  if (model1 == nullptr || model2 == nullptr) {
    std::cerr << "Create and build model failed." << std::endl;
    return -1;
  }

  std::thread thread1([&]() {
    auto generate_input_ret = GetInputsByTensorNameAndSetData(model1);
    if (generate_input_ret != mindspore::kSuccess) {
      std::cerr << "Model1 set input data error " << generate_input_ret << std::endl;
      return -1;
    }

    auto inputs = model1->GetInputs();
    auto outputs = model1->GetOutputs();
    auto predict_ret = model1->Predict(inputs, &outputs);
    if (predict_ret != mindspore::kSuccess) {
      std::cerr << "Model1 predict error " << predict_ret << std::endl;
      return -1;
    }
    std::cout << "Model1 predict success" << std::endl;
    return 0;
  });

  std::thread thread2([&]() {
    auto generate_input_ret = GetInputsByTensorNameAndSetData(model2);
    if (generate_input_ret != mindspore::kSuccess) {
      std::cerr << "Model2 set input data error " << generate_input_ret << std::endl;
      return -1;
    }

    auto inputs = model2->GetInputs();
    auto outputs = model2->GetOutputs();
    auto predict_ret = model2->Predict(inputs, &outputs);
    if (predict_ret != mindspore::kSuccess) {
      std::cerr << "Model2 predict error " << predict_ret << std::endl;
      return -1;
    }
    std::cout << "Model2 predict success" << std::endl;
    return 0;
  });

  thread1.join();
  thread2.join();

  // Get outputs data.
  // You can also get output through other methods,
  // and you can refer to GetOutputByTensorName() or GetOutputs().
  GetOutputsByNodeName(model1);
  GetOutputsByNodeName(model2);

  // Delete model.
  delete model1;
  delete model2;
  return 0;
}
```

### Mixed Precision Inference

MindSpore Lite supports mixed precision inference.
Users can set mixed precision information by calling the [LoadConfig](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore.html#loadconfig) API of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) after a model is created and before built.
The example of the config file is as follows:

```text
[execution_plan]
op_name1=data_type:float16
op_name2=data_type:float32
```

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L470) demonstrates how to infer model in the mixed precision:

```cpp
Status load_config_ret = model->LoadConfig(config_file_path);
if (load_config_ret != mindspore::kSuccess) {
  std::cerr << "Model load config error " << load_config_ret << std::endl;
  return -1;
}

Status build_ret = model->Build(graph_cell, context);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Model build error " << build_ret << std::endl;
  return -1;
}

auto inputs = model->GetInputs();
auto outputs = model->GetOutputs();
Status predict_ret = model->Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  std::cerr << "Model predict error " << predict_ret << std::endl;
  return -1;
}
```

### Sharing a Memory Pool

If there are multiple [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model), you can configure the same [Allocator](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Allocator.html#class-allocator) in [DeviceInfoContext](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_DeviceInfoContext.html#class-deviceinfocontext) to share the memory pool and reduce the memory size during running. The maximum memory size of the memory pool is `3 GB`, and the maximum memory size allocated each time is `2 GB`.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L546) demonstrates how to share the memory pool between two models:

```cpp
auto context1 = std::make_shared<mindspore::Context>();
if (context1 == nullptr) {
  std::cerr << "New context failed." << std::endl;
}
auto &device_list1 = context1->MutableDeviceInfo();
auto device_info1 = CreateCPUDeviceInfo();
if (device_info1 == nullptr) {
  std::cerr << "Create CPUDeviceInfo failed." << std::endl;
}
device_list1.push_back(device_info1);

auto model1 = new (std::nothrow) mindspore::Model();
if (model1 == nullptr) {
  std::cerr << "New Model failed." << std::endl;
}
auto build_ret = model1->Build(model_buf, size, mindspore::kMindIR, context1);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model failed." << std::endl;
}

auto context2 = std::make_shared<mindspore::Context>();
if (context2 == nullptr) {
  std::cerr << "New context failed." << std::endl;
}
auto &device_list2 = context2->MutableDeviceInfo();
auto device_info2 = CreateCPUDeviceInfo();
if (device_info2 == nullptr) {
  std::cerr << "Create CPUDeviceInfo failed." << std::endl;
}
// Use the same allocator to share the memory pool.
device_info2->SetAllocator(device_info1->GetAllocator());
device_list2.push_back(device_info2);

auto model2 = new (std::nothrow) mindspore::Model();
if (model2 == nullptr) {
  std::cerr << "New Model failed." << std::endl;
}
build_ret = model2->Build(model_buf, size, mindspore::kMindIR, context2);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model failed." << std::endl;
}
```

### Calling Back a Model During the Running Process

MindSpore Lite can pass two [MSKernelCallBack](https://www.mindspore.cn/lite/api/en/master/generate/typedef_mindspore_MSKernelCallBack-1.html) function pointers to [Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict) to call back a model for inference. Compared with common graph execution, callback execution can obtain additional information during the running process to help developers analyze performance and debug bugs. Additional information includes:

- Name of the running node
- Input and output tensors before the current node is inferred
- Input and output tensors after the current node is inferred

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L672) demonstrates how to define two callback functions as the pre-callback pointer and post-callback pointer and pass them to the Predict API for callback inference.

```cpp
// Definition of callback function before forwarding operator.
auto before_call_back = [](const std::vector<mindspore::MSTensor> &before_inputs,
                           const std::vector<mindspore::MSTensor> &before_outputs,
                           const mindspore::MSCallBackParam &call_param) {
  std::cout << "Before forwarding " << call_param.node_name_ << " " << call_param.node_type_ << std::endl;
  return true;
};
// Definition of callback function after forwarding operator.
auto after_call_back = [](const std::vector<mindspore::MSTensor> &after_inputs,
                          const std::vector<mindspore::MSTensor> &after_outputs,
                          const mindspore::MSCallBackParam &call_param) {
  std::cout << "After forwarding " << call_param.node_name_ << " " << call_param.node_type_ << std::endl;
  return true;
};

auto inputs = model->GetInputs();
auto outputs = model->GetOutputs();
auto predict_ret = model->Predict(inputs, &outputs, before_call_back, after_call_back);
if (predict_ret != mindspore::kSuccess) {
  std::cerr << "Predict error " << predict_ret << std::endl;
}
```

### Separating Graph Loading and Model Build

Use [Load](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#load) of [Serialization](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Serialization.html#class-serialization) to load [Graph](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Graph.html#class-graph) and use [Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build) of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) to build the model.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L282) demonstrates how to load graph and build model separately.

```cpp
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
  std::cerr << "New context failed." << std::endl;
}
auto &device_list = context->MutableDeviceInfo();
auto cpu_device_info = CreateCPUDeviceInfo();
if (cpu_device_info == nullptr) {
  std::cerr << "Create CPUDeviceInfo failed." << std::endl;
}
device_list.push_back(cpu_device_info);

// Load graph
mindspore::Graph graph;
auto load_ret = mindspore::Serialization::Load(model_buf, size, mindspore::kMindIR, &graph);
if (load_ret != mindspore::kSuccess) {
  std::cerr << "Load graph failed." << std::endl;
}

// Create model
auto model = new (std::nothrow) mindspore::Model();
if (model == nullptr) {
  std::cerr << "New Model failed." << std::endl;
  return nullptr;
}
// Build model
mindspore::GraphCell graph_cell(graph);
auto build_ret = model->Build(graph_cell, context);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model failed." << std::endl;
}
```

### Viewing Logs

If an exception occurs during inference, you can view logs to locate the fault. For the Android platform, use the `Logcat` command line to view the MindSpore Lite inference log information and use `MS_LITE` to filter the log information.

```bash
logcat -s "MS_LITE"
```

> For the iOS platform, does not support viewing logs temporarily.

### Obtaining the Version Number

MindSpore Lite provides the [Version](https://www.mindspore.cn/lite/api/en/master/generate/function_mindspore_Version-1.html#function-documentation) method to obtain the version number, which is included in the `include/api/types.h` header file. You can call this method to obtain the version number of MindSpore Lite.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L717) demonstrates how to obtain the version number of MindSpore Lite:

```cpp
#include "include/api/types.h"
std::string version = mindspore::Version();
```
