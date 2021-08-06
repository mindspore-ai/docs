# 使用C++接口执行推理

`Windows` `macOS` `Linux` `iOS` `Android` `C++` `推理应用` `模型加载` `数据准备` `中级` `高级`

MindSpore已经统一了端边云推理API，如您想继续使用MindSpore Lite独立API进行端侧推理，可以参考[此文档](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/runtime_cpp.html)。

<!-- TOC -->

- [使用C++接口执行推理](#使用c接口执行推理)
    - [概述](#概述)
    - [模型读取](#模型读取)
    - [创建配置上下文](#创建配置上下文)
        - [配置线程数](#配置线程数)
        - [配置线程亲和性](#配置线程亲和性)
        - [配置并行策略](#配置并行策略)
        - [配置使用GPU后端](#配置使用gpu后端)
        - [配置使用NPU后端](#配置使用npu后端)
        - [配置使用NNIE后端](#配置使用NNIE后端)
    - [模型创建加载与编译](#模型创建加载与编译)
    - [输入数据](#输入数据)
    - [执行推理](#执行推理)
    - [获取输出](#获取输出)
    - [内存释放](#内存释放)
    - [高级用法](#高级用法)
        - [输入维度Resize](#输入维度resize)
        - [Model并行](#Model并行)
        - [共享内存池](#共享内存池)
        - [回调运行](#回调运行)
        - [模型加载与编译独立调用流程](#模型加载与编译独立调用流程)
        - [查看日志](#查看日志)
        - [获取版本号](#获取版本号)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/use/runtime_cpp.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

通过[MindSpore Lite模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)转换成`.ms`模型后，即可在Runtime中执行模型的推理流程。本教程介绍如何使用[C++接口](https://www.mindspore.cn/lite/api/zh-CN/master/index.html)执行推理。

使用MindSpore Lite推理框架主要包括以下步骤：

1. 模型读取：从文件系统中读取由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)转换得到的`.ms`模型。
2. 创建配置上下文：创建配置上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)，保存需要的一些基本配置参数，用于指导模型编译和模型执行。
3. 模型创建、加载与编译：执行推理之前，需要调用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)的[Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build)接口进行模型加载和模型编译。模型加载阶段将文件缓存解析成运行时的模型。模型编译阶段主要进行算子选型调度、子图切分等过程，该阶段会耗费较多时间所以建议Model创建一次，编译一次，多次推理。
4. 输入数据：模型执行之前需要向`输入Tensor`中填充数据。
5. 执行推理：使用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)的[Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict)进行模型推理。
6. 获得输出：模型执行结束之后，可以通过`输出Tensor`得到推理结果。
7. 释放内存：无需使用MindSpore Lite推理框架时，需要释放已创建的[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)。

![img](../images/lite_runtime.png)

> 快速了解MindSpore Lite执行推理的完整调用流程，请参考[体验MindSpore Lite C++极简Demo](https://www.mindspore.cn/lite/docs/zh-CN/master/quick_start/quick_start_cpp.html)。

## 模型读取

通过MindSpore Lite进行模型推理时，需要从文件系统读取[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)转换得到的`.ms`模型文件。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L332)演示了从文件系统读取MindSpore Lite模型。

```cpp
// Read model file.
size_t size = 0;
char *model_buf = ReadFile(model_path, &size);
if (model_buf == nullptr) {
    std::cerr << "Read model file failed." << std::endl;
}
```

## 创建配置上下文

上下文会保存一些所需的基本配置参数，用于指导模型编译和模型执行，如果用户通过`new`创建[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)，不再需要时，需要用户通过`delete`释放。一般在创建编译完[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)后，[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)即可释放。

MindSpore Lite默认执行的后端是CPU，Context创建后调用[MutableDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mutabledeviceinfo)返回后端信息列表的引用，向列表中添加默认的[CPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#cpudeviceinfo)。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L250)演示了如何创建Context，配置默认的CPU后端，并设定CPU使能Float16推理。

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

> `MutableDeviceInfo`中第一个必须是CPU的[CPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#cpudeviceinfo), 第二个是GPU的[GPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#gpudeviceinfo)或者NPU的[KirinNPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#kirinnpudeviceinfo)。暂时不支持同时设置CPU，GPU和NPU三个后端。
>
> Float16需要CPU为ARM v8.2架构的机型才能生效，其他不支持的机型和x86平台会自动回退到Float32执行。
>
> 对于iOS设备,暂时只支持向`MutableDeviceInfo`添加CPU后端，且暂时不支持CPU后端Float16的执行。

[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)中包含的配置API如下：

### 配置线程数

[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)通过[SetThreadNum](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setthreadnum)配置线程数：

```cpp
// Configure the number of worker threads in the thread pool to 2, including the main thread.
context->SetThreadNum(2);
```

### 配置线程亲和性

[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)通过[SetThreadAffinity](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setthreadaffinity)配置线程和CPU绑定。如果参数是`int mode`，配置绑核策略，有效值为0-2，0为默认不绑核，1为优先绑大核，2为优先绑小核。如果参数是`const std::vector<int> &core_list`，配置绑核列表。同时配置时，core_list生效，mode不生效。

```cpp
// Configure the thread to be bound to the big core first.
// Valid value: 0: no affinities, 1: big cores first, 2: little cores first
context->SetThreadAffinity(1);
```

### 配置并行策略

[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)通过[SetEnableParallel](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setenableparallel)配置执行推理时是否支持并行。

```cpp
// Configure the inference supports parallel.
context->SetEnableParallel(true);
```

### 配置使用GPU后端

当需要执行的后端为CPU和GPU的异构推理时，需要同时设置[CPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#cpudeviceinfo)和[GPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#gpudeviceinfo)，配置后将会优先使用GPU推理。其中GpuDeviceInfo通过`SetEnableFP16`使能Float16推理。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L114)演示如何创建CPU与GPU异构推理后端，同时GPU也设定使能Float16推理：

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

auto gpu_device_info = std::make_shared<mindspore::GPUDeviceInfo>();
if (gpu_device_info == nullptr) {
  std::cerr << "New GPUDeviceInfo failed." << std::endl;
}
// If GPU device info is set. The preferred backend is GPU, which means, if there is a GPU operator, it will run on the GPU first, otherwise it will run on the CPU.
// GPU use float16 operator as priority.
gpu_device_info->SetEnableFP16(true);
// The GPU device context needs to be push_back into device_list to work.
device_list.push_back(gpu_device_info);
```

> 目前GPU的后端，在`arm64`上是基于OpenCL，支持Mali、Adreno的GPU，OpenCL版本为2.0。
>
> 具体配置为：
>
> CL_TARGET_OPENCL_VERSION=200
>
> CL_HPP_TARGET_OPENCL_VERSION=120
>
> CL_HPP_MINIMUM_OPENCL_VERSION=120
>
> 在`x86_64`上是基于TensorRT的GPU，TensorRT版本为6.0.1.5。`enable_float16_`属性是否设置成功取决于当前设备的[CUDA计算能力](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix)。

### 配置使用NPU后端

当需要执行的后端为CPU和NPU的异构推理时，需要同时设置[CPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#cpudeviceinfo)和[KirinNPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#kirinnpudeviceinfo)，配置后将会优先使用NPU推理，其中KirinNPUDeviceInfo通过`SetFrequency`来设置NPU频率。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L127)如何创建CPU与NPU异构推理后端，同时NPU频率设置为3。频率值默认为3，可设置为1（低功耗）、2（均衡）、3（高性能）、4（极致性能）：

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

auto npu_device_info = std::make_shared<mindspore::KirinNPUDeviceInfo>();
if (npu_device_info == nullptr) {
  std::cerr << "New KirinNPUDeviceInfo failed." << std::endl;
}
// NPU set frequency to be 3.
npu_device_info->SetFrequency(3);
// The NPU device context needs to be push_back into device_list to work.
device_list.push_back(npu_device_info);
```

### 配置使用NNIE后端

当需要执行的后端为CPU和NNIE的异构推理时，只需要按照[配置使用CPU后端](#创建配置上下文)的方法创建好Context即可，无需指定provider。

## 模型创建加载与编译

使用MindSpore Lite执行推理时，[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)是推理的主入口，通过Model可以实现模型加载、模型编译和模型执行。采用上一步创建得到的[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)，调用Model的复合[Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build)接口来实现模型加载与模型编译。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L265)演示了Model创建、加载与编译的过程：

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

> 创建并编译完[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)后，上一步创建得到的[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)即可释放。

## 输入数据

在模型执行前，需要获取到模型的输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)，将输入数据通过`memcpy`拷贝到模型的输入Tensor。可以通过MSTensor的[DataSize](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#datasize)方法来获取Tensor应该填入的数据大小，通过[DataType](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#datatype)方法来获取Tensor的数据类型，通过[MutableData](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mutabledata)方法来获取可写的指针。

MindSpore Lite提供两种方法来获取模型的输入Tensor。

1. 使用[GetInputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputbytensorname)方法，根据Tensor的名称来获取模型输入Tensor，下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L154)演示如何调用`GetInputByTensorName`获得输入Tensor并填充数据。

   ```cpp
   // Pre-processing of input data, convert input data format to NHWC.
   ...
   // Assume that the model has only one input tensor named graph_input-173.
   auto in_tensor = model->GetInputByTensorName("graph_input-173");
   if (in_tensor == nullptr) {
       std::cerr << "Input tensor is nullptr" << std::endl;
   }
   auto input_data = in_tensor.MutableData();
   if (input_data == nullptr) {
       std::cerr << "MallocData for inTensor failed." << std::endl;
   }
   memcpy(in_data, input_buf, data_size);
   // Users need to free input_buf.
   ```

2. 使用[GetInputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputs)方法，直接获取所有的模型输入Tensor的vector，下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L137)演示如何调用`GetInputs`获得输入Tensor并填充数据。

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

> MindSpore Lite的模型输入Tensor中的数据排布必须是`NHWC`。如果需要了解更多数据前处理过程，可参考基于JNI接口的Android应用开发中[编写端侧推理代码](https://www.mindspore.cn/lite/docs/zh-CN/master/quick_start/quick_start.html#id10)的第2步，将输入图片转换为传入MindSpore模型的Tensor格式。
>
> [GetInputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputs)和[GetInputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputbytensorname)方法返回的数据不需要用户释放。

## 执行推理

MindSpore Lite调用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)的[Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict)进行模型推理。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L355)演示调用`Predict`执行推理。

```cpp
auto inputs = model->GetInputs();
auto outputs = model->GetOutputs();
auto predict_ret = model->Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  std::cerr << "Predict error " << predict_ret << std::endl;
}
```

## 获取输出

MindSpore Lite在执行完推理后，就可以获取模型的推理结果。MindSpore Lite提供三种方法来获取模型的输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

1. 使用[GetOutputsByNodeName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputsbynodename)方法，根据模型输出节点的名称来获取模型输出Tensor中连接到该节点的Tensor的vector，下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L170)演示如何调用`GetOutputsByNodeName`获得输出Tensor。

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

2. 使用[GetOutputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputbytensorname)方法，根据模型输出Tensor的名称来获取对应的模型输出Tensor，下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L200)演示如何调用`GetOutputByTensorName`获得输出Tensor。

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

3. 使用[GetOutputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputs)方法，直接获取所有的模型输出Tensor的vector，下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L226)演示如何调用`GetOutputs`获得输出Tensor。

   ```cpp
   // Assume we have created a Model instance named model.
   auto out_tensors = model->GetOutputs();
   for (auto out_tensor : out_tensors) {
       // Post-processing the result data.
   }
   ```

> [GetOutputsByNodeName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputsbynodename)、[GetOutputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputbytensorname)和[GetOutputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getoutputs)方法返回的数据不需要用户释放。

## 内存释放

无需使用MindSpore Lite推理框架时，需要释放已经创建的Model，下列[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L370)演示如何在程序结束前进行内存释放。

```cpp
// Delete model.
// Assume that the variable of Model * is named model.
delete model;
```

## 高级用法

### 输入维度Resize

使用MindSpore Lite进行推理时，如果需要对输入的shape进行Resize，则可以在已完成创建[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)与模型编译[Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build)之后调用Model的[Resize](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#resize)接口，对输入的Tensor重新设置shape。

> 某些网络是不支持可变维度，会提示错误信息后异常退出，比如，模型中有MatMul算子，并且MatMul的一个输入Tensor是权重，另一个输入Tensor是输入时，调用可变维度接口会导致输入Tensor和权重Tensor的Shape不匹配，最终导致推理失败。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L321)演示如何对MindSpore Lite的输入Tensor进行Resize：

```cpp
// Assume we have created a Model instance named model.
auto inputs = model->GetInputs();
std::vector<int64_t> resize_shape = {1, 128, 128, 3};
// Assume the model has only one input,resize input shape to [1, 128, 128, 3]
std::vector<std::vector<int64_t>> new_shapes;
new_shapes.push_back(resize_shape);
return model->Resize(inputs, new_shapes);
```

### Model并行

MindSpore Lite支持多个[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)并行推理，每个Model的线程池和内存池都是独立的。但不支持多个线程同时调用单个Model的[Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict)接口。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L470)演示如何并行执行推理多个Model的过程：

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

### 共享内存池

如果存在多个[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)的情况，可以通过在[DeviceInfoContext](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#deviceinfocontext)中配置同一个[Allocator](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#allocator)，实现共享内存池来减少运行时内存大小。其中，内存池的内存总大小限制为`3G`，单次分配的内存限制为`2G`。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L546)演示如何在两个Model间共享内存池的功能：

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

### 回调运行

MindSpore Lite可以在调用[Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict)时，传入两个[MSKernelCallBack](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mskernelcallback)函数指针来回调推理模型，相比于一般的模型执行，回调运行可以在运行过程中获取额外的信息，帮助开发者进行性能分析、Bug调试等。额外的信息包括：

- 当前运行的节点名称
- 推理当前节点前的输入输出Tensor
- 推理当前节点后的输入输出Tensor

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L672)演示如何定义了两个回调函数作为前置回调指针和后置回调指针，传入到Predict接口进行回调推理。

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

### 模型加载与编译独立调用流程

模型加载与编译也可以分别调用[Serialization](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#serialization)的[Load](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#load)接口和[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)的[Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build)实现。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L282)演示模型加载与编译独立调用的流程：

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

### 查看日志

当推理出现异常的时候，可以通过查看日志信息来定位问题。针对Android平台，采用`Logcat`命令行工具查看MindSpore Lite推理的日志信息，并利用`MS_LITE` 进行筛选。

```bash
logcat -s "MS_LITE"
```

> 对iOS设备暂不支持日志查看。

### 获取版本号

MindSpore Lite提供了[Version](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#version)方法可以获取版本号，包含在`include/api/types.h`头文件中，调用该方法可以得到当前MindSpore Lite的版本号。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_cpp/main.cc#L717)演示如何获取MindSpore Lite的版本号：

```cpp
#include "include/api/types.h"
std::string version = mindspore::Version();
```
