# Using C++ Interface to Perform Cloud-side Inference

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/cloud_infer/runtime_cpp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

This tutorial describes how to perform cloud-side inference with MindSpore Lite by using the [C++ interface](https://www.mindspore.cn/lite/api/en/master/index.html).

MindSpore Lite cloud-side inference is supported to run in Linux environment deployment only. Ascend 310/310P/910, Nvidia GPU and CPU hardware backends are supported.

To experience the MindSpore Lite device-side inference process, please refer to the document [Using C++ Interface to Perform Cloud-side Inference](https://www.mindspore.cn/lite/docs/en/master/use/runtime_cpp.html).

Using the MindSpore Lite inference framework consists of the following main steps:

1. Model reading: Export MindIR model via MindSpore or get MindIR model by [model conversion tool](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/converter_tool.html).
2. Create a Configuration Context: Create a configuration context [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Converter.html) and save some basic configuration parameters used to guide model compilation and model execution.
3. Model loading and compilation: Before executing inference, you need to call Build interface of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) for model loading and model compilation. The model loading phase parses the file cache into a runtime model. The model compilation phase can take more time so it is recommended that the model be created once, compiled once and perform inference about multiple times.
4. Input data: The input data needs to be padded before the model can be executed.
5. Execute inference: Use Predict of [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) for model inference.

![img](../../images/lite_runtime.png)

## Preparation

1. The following code samples are from [using C++ interface to perform cloud-side inference sample code](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/runtime_cpp).

2. Export the MindIR model via MindSpore, or get the MindIR model by converting it with [model conversion tool](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/converter_tool.html) and copy it to the `mindspore/lite/examples/cloud_infer/runtime_cpp/model` directory. You can download the MobileNetV2 model file [mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir).

3. Download the Ascend, Nvidia GPU, CPU triplet MindSpore Lite cloud-side inference package `mindspore- lite-{version}-linux-{arch}.tar.gz` in the [official website](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) and save it to `mindspore/lite/examples/cloud_infer/runtime_cpp` directory.

## Creating Configuration Context

The context will save some basic configuration parameters used to guide model compilation and model execution.

The following sample code demonstrates how to create a Context.

```c++
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return nullptr;
}
auto &device_list = context->MutableDeviceInfo();
```

Return a reference to the list of backend information for specifying the running device via [MutableDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html). User-set device information is supported in `MutableDeviceInfo`, including [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_CPUDeviceInfo.html), [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_GPUDeviceInfo.html), [AscendDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_AscendDeviceInfo.html). The number of devices set can only be one of them currently.

### Configuring to Use the CPU Backend

When the backend to be executed is CPU, you need to set [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_CPUDeviceInfo.html) as the inference backend. Enable Float16 inference by `SetEnableFP16`.

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

Optionally, you can additionally set the number of threads, thread affinity, parallelism strategy and other features.

1. Configure the number of threads

    [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html) configure the number of threads via [SetThreadNum](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html):

    ```c++
    // Configure the number of worker threads in the thread pool to 2, including the main thread.
    context->SetThreadNum(2);
    ```

2. Configure thread affinity

    [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html) configure threads and CPU binding via [SetThreadAffinity](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html).
    Set the CPU binding list with the parameter `const std::vector<int> &core_list`.

    ```c++
    // Configure the thread to be bound to the core list.
    context->SetThreadAffinity({0,1});
    ```

3. Configure parallelism strategy

    [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html) configure the number of operator parallel inference at runtime via [SetInterOpParallelNum](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html).

    ```c++
    // Configure the inference supports parallel.
    context->SetInterOpParallelNum(2);
    ```

### Configuring Using GPU Backend

When the backend to be executed is GPU, you need to set [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_GPUDeviceInfo.html#class-gpudeviceinfo) as the inference backend. GPUDeviceInfo sets the device ID by `SetDeviceID` and enables Float16 inference by `SetEnableFP16`.

The following sample code demonstrates how to create a GPU inference backend while the device ID is set to 0:

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

> Whether the `SetEnableFP16` is set successfully depends on the [CUDA computing power] of the current device (https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix).

### Configuring Using Ascend Backend

When the backend to be executed is Ascend (Ascend 310/310P/910 are currently supported), you need to set [AscendDeviceInfo](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_AscendDeviceInfo.html#class-ascenddeviceinfo) as the inference backend. AscendDeviceInfo sets the device ID by `SetDeviceID`. Ascend enables Float16 precision by default, and the precision mode can be changed by `AscendDeviceInfo.SetPrecisionMode`.

The following sample code demonstrates how to create Ascend inference backend while the device ID is set to 0:

```c++
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
  return nullptr;
}
auto &device_list = context->MutableDeviceInfo();

// for Ascend 310/310P/910
auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
if (device_info == nullptr) {
  std::cerr << "New AscendDeviceInfo failed." << std::endl;
  return nullptr;
}
// Set Ascend 310/310P/910 device id.
device_info->SetDeviceID(device_id);
// The Ascend device context needs to be push_back into device_list to work.
device_list.push_back(gpu_device_info);
```

## Model Creation Loading and Compilation

When using MindSpore Lite to perform inference, [Model](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html#class-model) is the main entry point for inference. Model loading, model compilation and model execution is implemented through model. Using the [Context](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Context.html#class-context) created in the previous step, call the compound Build interface of Model to implement model loading and model compilation.

The following sample code demonstrates the process of model creation, loading and compilation:

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

## Inputting the Data

Before the model execution, the input data needs to be set, using the [GetInputs](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html) method, which directly gets all vectors of the model input Tensor. You can get the size of the data that the Tensor should fill in by the DataSize method of the [MSTensor](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_MSTensor.html). The data type of the Tensor can be obtained by the DataType. The input host data is set by SetData method.

There are currently two ways to specify input data:

1. By setting the input data via [SetData](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_MSTensor.html), copying between hosts can be avoided and the input data will eventually be copied directly to the inference device.

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

2. Copy the input data to the Tensor cache returned by [MutableData](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_MSTensor.html). It should be noted that if the data address has been set by `SetData`, `MutableData` will return the data address of `SetData`, and you need to call `SetData(nullptr)` first.

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

## Executing Inference

The Model.Predict interface is called to perform inference and subsequent processing of the returned output.

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

## Compilation and Execution

Set the environment variables as described in the [Environment Variables section in Quilk Start](https://www.mindspore.cn/lite/docs/en/master/quick_start/one_hour_introduction_cloud.html#environment-variables), and then compile the prograom as follows:

```bash
mkdir build && cd build
cmake ../
make
```

After successful compilation, you can get the `runtime_cpp` executable in the `build` directory. Execute program `runtime_cpp` to run the sample:

```bash
./runtime_cpp --model_path=../model/mobilenetv2.mindir --device_type=CPU
```

## Advanced Usage

### Dynamic Shape Input

Lite cloud-side inference framework supports dynamic shape input for models. GPU and Ascend hardware backend needs to be configured with dynamic input information during model conversion and model inference.

The configuration of dynamic input information is related to offline and online scenarios. For offline scenarios, the model conversion tool parameter `--NoFusion=false`, i.e. experiencing the hardware-related fusion and optimization. The generated MindIR model can only run on the corresponding hardware backend. For example, in Ascend 310 environment, if the model conversion tool specifies `---device=Ascend`, the generated model will only support running on Ascend 310. If `--device` is not specified, running on GPU and CPU is supported. For online scenarios, the loaded MindIR has not experienced hardware-related fusion and optimization, supports running on Ascend, GPU, and CPU. The model conversion tool parameter `--NoFusion=true`, or the MindSpore-exported MindIR model has not been processed by the conversion tool.

Ascend hardware backend offline scenarios require dynamic input information to be configured during the model conversion phase. Ascend hardware backend online scenarios, as well as GPU hardware backend offline and online scenarios, require dynamic input information to be configured during the model loading phase via the [LoadConfig](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore.html# loadconfig) interface.

An example configuration file loaded via `LoadConfig` is shown below:

```ini
[ascend_context]
input_shape=input_1:[-1,3,224,224]
dynamic_dims=[1~4],[8],[16]

[gpu_context]
input_shape=input_1:[-1,3,224,224]
dynamic_dims=[1~16]
opt_dims=[1]
```

The `[ascend_context]` and `[gpu_context]` act on the Ascend and GPU hardware backends, respectively.

1. Ascend and GPU hardware backends require dynamic input information for graph compilation and optimization, while CPU hardware backends do not require configuration of dynamic dimensional information.

2. `input_shape` is used to indicate the input shape information in the format `input_name1:[shape1];input_name2:[shape2]`. If there are dynamic inputs, the corresponding dimension needs to be set to -1. Multiple inputs are separated by the English semicolon `;`.

3. `dynamic_dims` is used to indicate the value range of the dynamic dimension, with multiple non-contiguous ranges of values separated by the comma `,`. In the above example, Ascend batch dimension values range in `1,2,3,4,8,16` and GPU batch dimension values range from 1 to 16. Ascend hardware backend with dynamic inputs are in multi-step mode, the larger the dynamic input range, the longer the model compilation time.

4. For the GPU hardware backend, additional configuration of `opt_dims` is required to indicate the optimal value in the `dynamic_dims` range.

5. If `input_shape` is configured as a static shape, `dynamic_dims` and `opt_dims` do not need to be configured.

Load the configuration file information via `LoadConfig` before the model `Build`:

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

In model inference, if the input to the model is dynamic and the input and output shape returned via `GetInputs` and `GetOutputs` may include -1, i.e., it is a dynamic shape,  the input shape needs to be specified via the [Resize](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_Model.html) interface. If the input Shape needs to change, for example, the `batch` dimension changes, the `Resize` interface needs to be called again to adjust the input Shape.

After calling the `Resize` interface, the shape of the Tensor in the called and subsequently called `GetInputs` and `GetOutputs` will be changed.

The following sample code demonstrates how to `Resize` the input Tensor of MindSpore Lite:

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

### Specifying Input and Output Host Memory

Specifies that device memory supports CPU, Ascend, and GPU hardware backends. The specified input host memory and the data in the cache will be copied directly to the device memory, and the data in the device memory will be copied directly to this cache for the specified output host memory. It avoids additional data copying between hosts and improves inference performance.

Input and output host memory can be specified separately or simultaneously by [SetData](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_MSTensor.html). It is recommended that the parameter `own_data` be false. When `own_data` is false, the user needs to maintain the life cycle of host memory and is responsible for the request and release of host memory. When the parameter `own_data` is true, the specified memory is freed at the MSTensor destruct.

1. Specify input host memory

    The values of input host memory are generally derived from the preprocessing results of C++ and Python on the host side.

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

2. Specify output host memory

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

### Specifying the Memory of the Input and Output Devices

Specifying device memory supports Ascend and GPU hardware backends. Specifying input and output device memory can avoid mutual copying from device to host memory, for example, the device memory input generated by chip dvpp preprocessing is directly used as input for model inference, avoiding preprocessing results copied from device memory to host memory and host results used as model inference input and re-copied to device before inference.

Sample memory for specified input and output devices can be found in [sample device memory](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/device_example_cpp).

Input and output device memory can be specified separately or simultaneously by [SetDeviceData](https://www.mindspore.cn/lite/api/en/master/generate/classmindspore_MSTensor.html). The user needs to maintain the device memory lifecycle and is responsible for device memory requests and releases.

1. Specify the input device memory

    In the sample, the value of the input device memory is copied from host, and the value of the general device memory comes from the preprocessing result of chip or the output of another model.

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

2. Specify the output device memory

    In the sample, the output device memory is copied to the host and prints the output. Generally the output device memory can be used as input for other models.

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
