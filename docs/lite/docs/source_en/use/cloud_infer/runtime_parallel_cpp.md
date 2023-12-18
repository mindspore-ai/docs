# Using C++ Interface to Perform Concurrent Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/lite/docs/source_en/use/cloud_infer/runtime_parallel_cpp.md)

## Overview

MindSpore Lite provides multi-model concurrent inference interface [ModelParallelRunner](https://www.mindspore.cn/lite/api/en/r2.3/api_java/model_parallel_runner.html). Multi model concurrent inference now supports Atlas 200/300/500 inference product, Atlas inference series (with Ascend 310P AI processor), Atlas training series, Nvidia GPU and CPU backends.

After exporting the `mindir` model by MindSpore or converting it by [model conversion tool](https://www.mindspore.cn/lite/docs/en/r2.3/use/cloud_infer/converter_tool.html) to obtain the `mindir` model, the concurrent inference process of the model can be executed in Runtime. This tutorial describes how to perform concurrent inference with multiple modes by using the [C++ interface](https://www.mindspore.cn/lite/api/en/r2.3/index.html).

To use the MindSpore Lite concurrent inference framework, perform the following steps:

1. Create a configuration item: Create a multi-model concurrent inference configuration item [RunnerConfig](https://www.mindspore.cn/lite/api/en/r2.3/generate/classmindspore_RunnerConfig.html), which is used to configure multiple model concurrency.
2. Initialization: initialization before multi-model concurrent inference.
3. Execute concurrent inference: Use the Predict interface of ModelParallelRunner to perform concurrent inference on multiple models.
4. Release memory: When you do not need to use the MindSpore Lite concurrent inference framework, you need to release the ModelParallelRunner and related Tensors you created.

![](../images/server_inference.png)

## Preparation

1. The following code samples are from [Sample code for performing cloud-side inference by C++ interface](https://gitee.com/mindspore/mindspore/tree/r2.3/mindspore/lite/examples/cloud_infer/quick_start_parallel_cpp).

2. Export the MindIR model via MindSpore, or get the MindIR model by converting it with [model conversion tool](https://www.mindspore.cn/lite/docs/en/r2.3/use/cloud_infer/converter_tool.html) and copy it to the `mindspore/lite/examples/cloud_infer/quick_start_parallel_cpp/model` directory, and you can download the MobileNetV2 model file [mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir).

3. Download the Ascend, Nvidia GPU, CPU triplet MindSpore Lite cloud-side inference package `mindspore- lite-{version}-linux-{arch}.tar.gz` and save it to `mindspore/lite/examples/cloud_infer/quick_start_parallel_cpp` directory.

## Create configuration

The [configuration item](https://www.mindspore.cn/lite/api/en/r2.3/generate/classmindspore_RunnerConfig.html) will save some basic configuration parameters required for concurrent inference, which are used to guide the number of concurrent models, model compilation and model execution.

The following sample code from main.cc demonstrates how to create a RunnerConfig and configure the number of workers for concurrent inference:

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

> For details on the configuration method of Context, see [Context](https://www.mindspore.cn/lite/api/en/r2.3/generate/classmindspore_Context.html).
>
> Multi-model concurrent inference currently only supports [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.3/generate/classmindspore_CPUDeviceInfo.html), [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.3/generate/classmindspore_GPUDeviceInfo.html), and [AscendDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.3/generate/classmindspore_AscendDeviceInfo.html) several different hardware backends. When setting the GPU backend, you need to set the GPU backend first and then the CPU backend, otherwise it will report an error and exit.
>
> Multi-model concurrent inference does not support FP32-type data inference. Binding cores only supports no core binding or binding large cores. It does not support the parameter settings of the bound cores, and does not support configuring the binding core list.
>
> For large models, when using the model buffer to load and compile, you need to set the path of the weight file separately, sets the model path through [SetConfigInfo](https://www.mindspore.cn/lite/api/en/r2.3/generate/classmindspore_RunnerConfig.html) interface, where `section` is `model_File` , `key` is `mindir_path`; When using the model path to load and compile, you do not need to set other parameters. The weight parameters will be automatically read.

## Initialization

When using MindSpore Lite to execute concurrent inference, ModelParallelRunner is the main entry of concurrent inference. Through ModelParallelRunner, you can initialize and execute concurrent inference. Use the RunnerConfig created in the previous step and call the init interface of ModelParallelRunner to initialize ModelParallelRunner.

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

## Executing Concurrent Inference

MindSpore Lite calls the Predict interface of ModelParallelRunner for model concurrent inference.

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

## Memory Release

When you do not need to use the MindSpore Lite inference framework, you need to release the created ModelParallelRunner.

```cpp
// Delete model runner.
delete model_runner;
```
