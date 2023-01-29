# Using C++ Interface to Parallel Inference

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/lite/docs/source_en/use/runtime_server_inference_cpp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore Lite provides multi-model concurrent inference interface [ModelParallelRunner](https://www.mindspore.cn/lite/api/en/r2.0.0-alpha/generate/classmindspore_ModelParallelRunner.html). Multi model concurrent inference now supports CPU and GPU backend.

> For a quick understanding of the complete calling process of MindSpore Lite executing concurrent reasoning, please refer to [Experience C++ Minimalist Concurrent Reasoning Demo](https://www.mindspore.cn/lite/docs/en/r2.0.0-alpha/quick_start/quick_start_server_inference_cpp.html).

After the model is converted into a `.ms` model by using the MindSpore Lite model [conversion tool](https://www.mindspore.cn/lite/docs/en/r2.0.0-alpha/use/converter_tool.html), the inference process can be performed in Runtime. For details, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/r2.0.0-alpha/use/converter_tool.html). This tutorial describes how to use the [C++ API](https://www.mindspore.cn/lite/api/en/r2.0.0-alpha/index.html) to perform inference.

To use the MindSpore Lite parallel inference framework, perform the following steps:

1. Create a configuration item: Create a multi-model concurrent inference configuration item [RunnerConfig](https://www.mindspore.cn/lite/api/en/r2.0.0-alpha/generate/classmindspore_RunnerConfig.html), which is used to configure multiple model concurrency.
2. Initialization: initialization before multi-model concurrent inference.
3. Execute concurrent inference: Use the Predict interface of ModelParallelRunner to perform concurrent inference on multiple models.
4. Release memory: When you do not need to use the MindSpore Lite concurrent inference framework, you need to release the ModelParallelRunner and related Tensors you created.

## Create configuration

The [configuration item](https://www.mindspore.cn/lite/api/en/r2.0.0-alpha/generate/classmindspore_RunnerConfig.html) will save some basic configuration parameters required for concurrent reasoning, which are used to guide the number of concurrent models, model compilation and model execution.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/mindspore/lite/examples/quick_start_server_inference_cpp/main.cc#L135) demonstrates how to create a RunnerConfig and configure the number of workers for concurrent reasoning:

```cpp
// Create and init context, add CPU device info
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
if (runner_config == nullptr) {
  std::cerr << "runner config is nullptr." << std::endl;
  return -1;
}
runner_config->SetContext(context);
runner_config->SetWorkersNum(kNumWorkers);
```

> For details on the configuration method of Context, see [Context](https://www.mindspore.cn/lite/api/en/r2.0.0-alpha/generate/classmindspore_Context.html).
>
> Multi-model concurrent inference currently only supports [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.0.0-alpha/generate/classmindspore_CPUDeviceInfo.html) and [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.0.0-alpha/generate/classmindspore_GPUDeviceInfo.html) two different hardware backends. When setting the GPU backend, you need to set the GPU backend first and then the CPU backend, otherwise it will report an error and exit.
>
> Multi-model concurrent inference does not support FP32-type data inference. Binding cores only supports no core binding or binding large cores. It does not support the parameter settings of the bound cores, and does not support configuring the binding core list.

## Initialization

When using MindSpore Lite to execute concurrent inference, ModelParallelRunner is the main entry of concurrent reasoning. Through ModelParallelRunner, you can initialize and execute concurrent reasoning. Use the RunnerConfig created in the previous step and call the init interface of ModelParallelRunner to initialize ModelParallelRunner.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/mindspore/lite/examples/quick_start_server_inference_cpp/main.cc#L155) demonstrates how to call `Predict` to execute reasoning:

```cpp
// Build model
auto build_ret = model_runner->Init(model_path, runner_config);
if (build_ret != mindspore::kSuccess) {
delete model_runner;
  std::cerr << "Build model error " << build_ret << std::endl;
  return -1;
}
```

> For Initialization of ModelParallelRunner, you do not need to set the RunnerConfig configuration parameters, and the default parameters will be used for concurrent inference of multiple models.

## Execute concurrent reasoning

MindSpore Lite calls the Predict interface of ModelParallelRunner for model concurrent inference.

The following [main.cc](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/mindspore/lite/examples/quick_start_server_inference_cpp/main.cc#L189) demonstrates how to call `Predict` to execute inference.

```cpp
// Model Predict
auto predict_ret = model_runner->Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  delete model_runner;
  std::cerr << "Predict error " << predict_ret << std::endl;
  return -1;
}
```

> It is recommended to use GetInputs and GetOutputs to obtain the inputs and outputs of the Predict interface. Users can set the memory address of the data and Shape-related information through SetData.

## Memory release

When you do not need to use the MindSpore Lite reasoning framework, you need to release the created ModelParallelRunner. The following [main.cc](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/mindspore/lite/examples/quick_start_server_inference_cpp/main.cc#L220) demonstrates how to free memory before the end of the program.

```cpp
// Delete model runner.
delete model_runner;
```
