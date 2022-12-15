# 使用C++接口执行云侧推理

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/use/cloud_infer/runtime_cpp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

本教程介绍如何使用[C++接口](https://www.mindspore.cn/lite/api/zh-CN/master/index.html)执行MindSpore Lite云侧推理。

MindSpore Lite云侧推理仅支持在Linux环境部署运行。支持Ascend310、Ascend310P、Nvidia GPU和CPU硬件后端。

如需体验MindSpore Lite端侧推理流程，请参考文档[使用C++接口执行端侧推理](https://www.mindspore.cn/lite/docs/zh-CN/master/use/runtime_cpp.html)。

使用MindSpore Lite推理框架主要包括以下步骤：

1. 模型读取：通过MindSpore导出MindIR模型，或者由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)转换获得MindIR模型。
2. 创建配置上下文：创建配置上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)，保存需要的一些基本配置参数，用于指导模型编译和模型执行。
3. 模型加载与编译：执行推理之前，需要调用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)的[Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build-3)接口进行模型加载和模型编译。模型加载阶段将文件缓存解析成运行时的模型。模型编译阶段会耗费较多时间所以建议Model创建一次，编译一次，多次推理。
4. 输入数据：模型执行之前需要填充输入数据。
5. 执行推理：使用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)的[Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict)进行模型推理。

![img](../../images/lite_runtime.png)

## 准备工作

1. 以下代码样例来自于[使用C++接口执行云侧推理示例代码](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/runtime_cpp)。

2. 通过MindSpore导出MindIR模型，或者由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)转换获得MindIR模型，并将其拷贝到`mindspore/lite/examples/cloud_infer/runtime_cpp/model`目录，可以下载MobileNetV2模型文件[mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir)。

3. 从[官网](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)下载Ascend、Nvidia GPU、CPU三合一的MindSpore Lite云侧推理包`mindspore-lite-{version}-linux-{arch}.tar.gz`，并存放到`mindspore/lite/examples/cloud_infer/runtime_cpp`目录。

## 创建配置上下文

上下文会保存一些所需的基本配置参数，用于指导模型编译和模型执行。

下面示例代码演示了如何创建Context。

```c++
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
}
auto &device_list = context->MutableDeviceInfo();
```

通过[MutableDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mutabledeviceinfo)返回后端信息列表的引用，指定运行的设备。`MutableDeviceInfo`中支持用户设置设备信息，包括[CPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#cpudeviceinfo)、[GPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#gpudeviceinfo)、[AscendDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#ascenddeviceinfo)。设置的设备个数当前只能为其中一个。

### 配置使用CPU后端

当需要执行的后端为CPU时，需要设置[CPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#cpudeviceinfo)为推理后端。通过`SetEnableFP16`使能Float16推理。

```c++
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
  std::cerr << "New context failed." << std::endl;
  return nullptr;
}
auto &device_list = context->MutableDeviceInfo();
auto cpu_device_info = std::make_shared<mindspore::CPUDeviceInfo>();
if (cpu_device_info == nullptr) {
  std::cerr << "New CPUDeviceInfo failed." << std::endl;
  return nullptr;
}
// CPU use float16 operator as priority.
cpu_device_info->SetEnableFP16(true);
device_list.push_back(cpu_device_info);
```

可选择性地额外设置线程数、线程亲和性、并行策略等特性。

1. 配置线程数

    [Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)通过[SetThreadNum](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setthreadnum)配置线程数：

    ```c++
    // Configure the number of worker threads in the thread pool to 2, including the main thread.
    context->SetThreadNum(2);
    ```

2. 配置线程亲和性

    [Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)通过[SetThreadAffinity](hhttps://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setthreadaffinity-1)配置线程和CPU绑定。
    通过参数`const std::vector<int> &core_list`设置绑核列表。

    ```c++
    // Configure the thread to be bound to the core list.
    context->SetThreadAffinity({0,1});
    ```

3. 配置并行策略

    [Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)通过[SetInterOpParallelNum](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setinteropparallelnum)设置运行时的算子并行推理数目。

    ```c++
    // Configure the inference supports parallel.
    context->SetInterOpParallelNum(2);
    ```

### 配置使用GPU后端

当需要执行的后端为GPU时，需要设置[GPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#gpudeviceinfo)为推理后端。其中GPUDeviceInfo通过`SetDeviceID`来设置设备ID，通过`SetEnableFP16`使能Float16推理。

下面示例代码演示如何创建GPU推理后端，同时设备ID设置为0：

```c++
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
  return nullptr;
}
auto &device_list = context->MutableDeviceInfo();

auto gpu_device_info = std::make_shared<mindspore::GPUDeviceInfo>();
if (gpu_device_info == nullptr) {
  std::cerr << "New GPUDeviceInfo failed." << std::endl;
  return nullptr;
}
// Set NVIDIA device id.
gpu_device_info->SetDeviceID(0);
// GPU use float16 operator as priority.
gpu_device_info->SetEnableFP16(true);
// The GPU device context needs to be push_back into device_list to work.
device_list.push_back(gpu_device_info);
```

> `SetEnableFP16`属性是否设置成功取决于当前设备的[CUDA计算能力](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix)。

### 配置使用Ascend后端

当需要执行的后端为Ascend时(目前支持Ascend 310和Ascend 310P)，需要设置[AscendDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#ascenddeviceinfo)为推理后端。其中AscendDeviceInfo通过`SetDeviceID`来设置设备ID。Ascend默认使能Float16精度，可通过`AscendDeviceInfo.SetPrecisionMode`更改精度模式。

下面示例代码演示如何创建Ascend推理后端，同时设备ID设置为0：

```c++
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
  return nullptr;
}
auto &device_list = context->MutableDeviceInfo();

// for Ascend 310, 310P
auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
if (device_info == nullptr) {
  std::cerr << "New AscendDeviceInfo failed." << std::endl;
  return nullptr;
}
// Set Ascend 310/310P device id.
device_info->SetDeviceID(device_id);
// The Ascend device context needs to be push_back into device_list to work.
device_list.push_back(gpu_device_info);
```

## 模型创建加载与编译

使用MindSpore Lite执行推理时，[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#model)是推理的主入口，通过Model可以实现模型加载、模型编译和模型执行。采用上一步创建得到的[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)，调用Model的复合[Build](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#build-3)接口来实现模型加载与模型编译。

下面示例代码演示了Model创建、加载与编译的过程：

```c++
std::shared_ptr<mindspore::Model> BuildModel(const std::string &model_path, const std::string &device_type,
                                             int32_t device_id) {
  // Create and init context, add CPU device info
  auto context = std::make_shared<mindspore::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return nullptr;
  }
  auto &device_list = context->MutableDeviceInfo();
  std::shared_ptr<mindspore::DeviceInfoContext> device_info = nullptr;
  if (device_type == "CPU") {
    device_info = CreateCPUDeviceInfo();
  } else if (device_type == "GPU") {
    device_info = CreateGPUDeviceInfo(device_id);
  } else if (device_type == "Ascend") {
    device_info = CreateAscendDeviceInfo(device_id);
  }
  if (device_info == nullptr) {
    std::cerr << "Create " << device_type << "DeviceInfo failed." << std::endl;
    return nullptr;
  }
  device_list.push_back(device_info);

  // Create model
  auto model = std::make_shared<mindspore::Model>();
  if (model == nullptr) {
    std::cerr << "New Model failed." << std::endl;
    return nullptr;
  }
  // Build model
  auto build_ret = model->Build(model_path, mindspore::kMindIR, context);
  if (build_ret != mindspore::kSuccess) {
    std::cerr << "Build model failed." << std::endl;
    return nullptr;
  }
  return model;
}
```

## 输入数据

在模型执行前，需要设置输入数据，使用[GetInputs](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#getinputs)方法，直接获取所有的模型输入Tensor的vector。可以通过MSTensor的[DataSize](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#datasize)方法来获取Tensor应该填入的数据大小，通过[DataType](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#datatype)方法来获取Tensor的数据类型，通过[SetData](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setdata)方法设置输入host数据。

当前有两种指定输入数据的方式：

1. 通过[SetData](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setdata-1)设置输入数据，可以避免host之间的拷贝，输入数据将最终直接拷贝到推理设备上。

    ```c++
    int SetTensorHostData(std::vector<mindspore::MSTensor> *tensors, std::vector<MemBuffer> *buffers) {
      if (!tensors || !buffers) {
        std::cerr << "Argument tensors or buffers cannot be nullptr" << std::endl;
        return -1;
      }
      if (tensors->size() != buffers->size()) {
        std::cerr << "tensors size " << tensors->size() << " != "
                  << " buffers size " << buffers->size() << std::endl;
        return -1;
      }
      for (size_t i = 0; i < tensors->size(); i++) {
        auto &tensor = (*tensors)[i];
        auto &buffer = (*buffers)[i];
        if (tensor.DataSize() != buffer.size()) {
          std::cerr << "Tensor data size " << tensor.DataSize() << " != buffer size " << buffer.size() << std::endl;
          return -1;
        }
        // set tensor data, and the memory should be freed by user
        tensor.SetData(buffer.data(), false);
        tensor.SetDeviceData(nullptr);
      }
      return 0;
    }

      auto inputs = model->GetInputs();
      // Set the input data of the model, this inference input will be copied directly to the device.
      SetTensorHostData(&inputs, &input_buffer);
    ```

2. 将输入数据拷贝到[MutableData](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mutabledata)返回的Tensor缓存中。注意，如果已通过`SetData`设置过数据地址，则`MutableData`返回的将时`SetData`的数据地址，此时需要先调用`SetData(nullptr)`。

    ```c++
    int CopyTensorHostData(std::vector<mindspore::MSTensor> *tensors, std::vector<MemBuffer> *buffers) {
      for (size_t i = 0; i < tensors->size(); i++) {
        auto &tensor = (*tensors)[i];
        auto &buffer = (*buffers)[i];
        if (tensor.DataSize() != buffer.size()) {
          std::cerr << "Tensor data size " << tensor.DataSize() << " != buffer size " << buffer.size() << std::endl;
          return -1;
        }
        auto dst_mem = tensor.MutableData();
        if (dst_mem == nullptr) {
          std::cerr << "Tensor MutableData return nullptr" << std::endl;
          return -1;
        }
        memcpy(tensor.MutableData(), buffer.data(), buffer.size());
      }
      return 0;
    }
      auto inputs = model->GetInputs();
      // Set the input data of the model, copy data to the tensor buffer of Model.GetInputs.
      CopyTensorHostData(&inputs, &input_buffer);
    ```

## 执行推理

调用[Model.Predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#predict)接口执行推理，并对返回的输出结果进行后续处理。

```c++
int SpecifyInputDataExample(const std::string &model_path, const std::string &device_type, int32_t device_id,
                            int32_t batch_size) {
  auto model = BuildModel(model_path, device_type, device_id);
  if (model == nullptr) {
    std::cerr << "Create and build model failed." << std::endl;
    return -1;
  }
  auto inputs = model->GetInputs();
  // InferenceApp is user-defined code. Users need to obtain inputs and process outputs based on
  // the actual situation.
  InferenceApp app;
  // Obtain inputs. The input data for inference may come from the preprocessing result.
  auto &input_buffer = app.GetInferenceInputs(inputs);
  if (input_buffer.empty()) {
    return -1;
  }
  // Set the input data of the model, this inference input will be copied directly to the device.
  SetTensorHostData(&inputs, &input_buffer);

  std::vector<mindspore::MSTensor> outputs;
  auto predict_ret = model->Predict(inputs, &outputs);
  if (predict_ret != mindspore::kSuccess) {
    std::cerr << "Predict error " << predict_ret << std::endl;
    return -1;
  }
  // Process outputs.
  app.OnInferenceResult(outputs);
  return 0;
}
```

## 高级用法

### 动态shape输入

Lite云侧推理框架支持动态shape输入的模型，GPU和Ascend硬件后端，需要在模型转换和模型推理时配置动态输入信息。

动态输入信息的配置与离线和在线场景有关。离线场景，模型转换工具参数`--NoFusion=false`，即经历和硬件相关的融合和优化，产生的MindIR模型仅能在对应硬件后端上运行，比如模型转换工具指定`--device=Ascend310`，则产生的模型仅支持在Ascend310上运行，如果未指定`--device`，则支持在GPU和CPU上运行。在线场景，加载的MindIR没有经历和硬件相关的融合和优化，支持在Ascend、GPU和CPU上运行，模型转换工具参数`--NoFusion=true`，或MindSpore导出的MindIR模型没有经过转换工具处理。

Ascend硬件后端离线场景下，需要在模型转换阶段配置动态输入信息。Ascend硬件后端在线场景下，以及GPU硬件后端离线和在线场景下，需要在模型加载阶段通过[LoadConfig](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#loadconfig)接口配置动态输入信息。

通过`LoadConfig`加载的配置文件示例如下所示：

```ini
[ascend_context]
input_shape=input_1:[-1,3,224,224]
dynamic_dims=[1~4],[8],[16]

[gpu_context]
input_shape=input_1:[-1,3,224,224]
dynamic_dims=[1~16]
opt_dims=[1]
```

`[ascend_context]`和`[gpu_context]`分别作用于Ascend和GPU硬件后端。

> Ascend和GPU硬件后端需要通过动态输入信息进行图的编译和优化，CPU硬件后端不需要配置动态维度信息。
>
> `input_shape`用于指示输入shape信息，格式为`input_name1:[shape1];input_name2:[shape2]`，如果有动态输入，则需要将相应的维度设定为-1,多个输入通过英文分号`;`隔开。
>
> `dynamic_dims`用于指示动态维度的值范围，多个非连续的值范围通过英文逗号`,`隔开。上例子中，Ascend的batch维度值范围为`1,2,3,4,8,16`，GPU的batch维度值范围为1到16。Ascend硬件后端，动态输入为多档模式，动态输入范围越大，模型编译时间越长。
>
> 对于GPU硬件后端，需要额外配置`opt_dims`用于指示`dynamic_dims`范围中最优的值。
>
> 如果`input_shape`配置的为静态shape，则不需要配置`dynamic_dims`和`opt_dims`。

在模型`Build`前，通过`LoadConfig`加载配置文件信息：

```c++
  // Create model
  auto model = std::make_shared<mindspore::Model>();
  if (model == nullptr) {
    std::cerr << "New Model failed." << std::endl;
    return nullptr;
  }
  if (!config_file.empty()) {
    if (model->LoadConfig(config_file) != mindspore::kSuccess) {
      std::cerr << "Failed to load config file " << config_file << std::endl;
      return nullptr;
    }
  }
  // Build model
  auto build_ret = model->Build(model_path, mindspore::kMindIR, context);
  if (build_ret != mindspore::kSuccess) {
    std::cerr << "Build model failed." << std::endl;
    return nullptr;
  }
```

在模型推理时，如果模型的输入是动态的，通过`GetInputs`和`GetOutputs`返回的输入输出shape可能包括-1，即为动态shape，则需要通过[Resize](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#resize)接口指定输入shape。如果输入Shape需要发生变化，比如`batch`维度发生变化，则需要重新调用`Resize`接口调整输入shape。

调用`Resize`接口后，已调用和后续调用的`GetInputs`和`GetOutputs`中的Tensor的shape将发生变化。

下面示例代码演示如何对MindSpore Lite的输入Tensor进行`Resize`：

```c++
int ResizeModel(std::shared_ptr<mindspore::Model> model, int32_t batch_size) {
  std::vector<std::vector<int64_t>> new_shapes;
  auto inputs = model->GetInputs();
  for (auto &input : inputs) {
    auto shape = input.Shape();
    shape[0] = batch_size;
    new_shapes.push_back(shape);
  }
  if (model->Resize(inputs, new_shapes) != mindspore::kSuccess) {
    std::cerr << "Failed to resize to batch size " << batch_size << std::endl;
    return -1;
  }
  return 0;
}
```

### 指定输入输出host内存

指定设备内存支持CPU、Asend和GPU硬件后端。指定的输入host内存，缓存中的数据将直接拷贝到设备（device）内存上，指定的输出host内存，设备（device）内存的数据将直接拷贝到这块缓存中。避免了额外的host之间的数据拷贝，提升推理性能。

通过[SetData](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setdata)可单独或者同时指定输入和输出host内存。建议参数`own_data`为false，当`own_data`为false，用户需要维护host内存的生命周期，负责host内存的申请和释放。当参数`own_data`为true时，在MSTensor析构时释放指定的内存。

1. 指定输入host内存

    输入host内存的值，一般来源于host侧的C++、Python等预处理的结果。

    ```c++
      std::vector<void *> host_buffers;
      // ... get host buffer from preprocessing etc.
      // Get Input
      auto inputs = model->GetInputs();
      for (size_t i = 0; i < tensors.size(); i++) {
        auto &tensor = tensors[i];
        auto host_data = host_buffers[i];
        tensor.SetData(host_data, false);
        tensor.SetDeviceData(nullptr);
      }

      std::vector<mindspore::MSTensor> outputs;
      if (Predict(model, inputs, &outputs) != 0) {
        return -1;
      }
    ```

2. 指定输出host内存

    ```c++
      // Get Output from model
      auto outputs = model->GetOutputs();
      std::vector<void *> output_buffers;
      ResourceGuard output_device_rel([&output_buffers]() {
        for (auto &item : output_buffers) {
          free(item);
        }
      });
      for (auto &tensor : outputs) {
        auto buffer = malloc(tensor.DataSize());
        tensor.SetData(buffer, false);
        tensor.SetDeviceData(nullptr);
        output_buffers.push_back(buffer); // for free
      }
      if (Predict(model, inputs, &outputs) != 0) {
        return -1;
      }
    ```

### 指定输入输出设备（device）内存

指定设备内存支持Asend和GPU硬件后端。指定输入输出设备内存可以避免device到host内存之间的相互拷贝，比如经过芯片dvpp预处理产生的device内存输入直接作为模型推理的输入，避免预处理结果从device内存拷贝到host内存，host结果作为模型推理输入，推理前重新拷贝到device上。

指定输入输出设备内存样例可参考[设备内存样例](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/device_example_cpp)。

通过[SetDeviceData](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#setdevicedata)可单独或者同时指定输入和输出设备内存。用户需要维护设备内存的生命周期，负责设备内存的申请和释放。

1. 指定输入设备内存

    样例中，输入设备内存的值拷贝自host，一般设备内存的值来自于芯片预处理的结果或另一个模型的输出。

    ```c++
    int SetDeviceData(std::vector<mindspore::MSTensor> tensors, const std::vector<uint8_t *> &host_data_buffer,
                      std::vector<void *> *device_buffers) {
      for (size_t i = 0; i < tensors.size(); i++) {
        auto &tensor = tensors[i];
        auto host_data = host_data_buffer[i];
        auto data_size = tensor.DataSize();
        if (data_size == 0) {
          std::cerr << "Data size cannot be 0, tensor shape: " << ShapeToString(tensor.Shape()) << std::endl;
          return -1;
        }
        auto device_data = MallocDeviceMemory(data_size);
        if (device_data == nullptr) {
          std::cerr << "Failed to alloc device data, data size " << data_size << std::endl;
          return -1;
        }
        device_buffers->push_back(device_data);
        if (CopyMemoryHost2Device(device_data, data_size, host_data, data_size) != 0) {
          std::cerr << "Failed to copy data to device, data size " << data_size << std::endl;
          return -1;
        }
        tensor.SetDeviceData(device_data);
        tensor.SetData(nullptr, false);
      }
      return 0;
    }

      // Get Input
      auto inputs = model->GetInputs();
      std::vector<void *> device_buffers;
      ResourceGuard device_rel([&device_buffers]() {
        for (auto &item : device_buffers) {
          FreeDeviceMemory(item);
        }
      });
      SetDeviceData(inputs, host_buffers, &device_buffers);
      std::vector<mindspore::MSTensor> outputs;
      if (Predict(model, inputs, &outputs) != 0) {
        return -1;
      }
    ```

2. 指定输出设备内存

    样例中，输出设备内存拷贝到host打印输出，一般输出设备内存可作为其他模型的输入。

    ```c++
    int SetOutputDeviceData(std::vector<mindspore::MSTensor> tensors, std::vector<void *> *device_buffers) {
      for (size_t i = 0; i < tensors.size(); i++) {
        auto &tensor = tensors[i];
        auto data_size = tensor.DataSize();
        if (data_size == 0) {
          std::cerr << "Data size cannot be 0, tensor shape: " << ShapeToString(tensor.Shape()) << std::endl;
          return -1;
        }
        auto device_data = MallocDeviceMemory(data_size);
        if (device_data == nullptr) {
          std::cerr << "Failed to alloc device data, data size " << data_size << std::endl;
          return -1;
        }
        device_buffers->push_back(device_data);
        tensor.SetDeviceData(device_data);
        tensor.SetData(nullptr, false);
      }
      return 0;
    }

      // Get Output from model
      auto outputs = model->GetOutputs();
      std::vector<void *> output_device_buffers;
      ResourceGuard output_device_rel([&output_device_buffers]() {
        for (auto &item : output_device_buffers) {
          FreeDeviceMemory(item);
        }
      });
      if (SetOutputDeviceData(outputs, &output_device_buffers) != 0) {
        std::cerr << "Failed to set output device data" << std::endl;
        return -1;
      }
      if (Predict(model, inputs, &outputs) != 0) {
        return -1;
      }
    ```
