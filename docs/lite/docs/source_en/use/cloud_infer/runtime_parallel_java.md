# Using Java Interface to Perform Concurrent Inference

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/cloud_infer/runtime_parallel_java.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore Lite provides multi-model concurrent inference interface [ModelParallelRunner](https://www.mindspore.cn/lite/api/en/master/api_java/model_parallel_runner.html#modelparallelrunner). Multi model concurrent inference now supports Ascend310, Ascend310P, Nvidia GPU and CPU backends.

After exporting the `mindir` model by MindSpore or converting it by [model conversion tool](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/converter_tool.html) to obtain the `mindir` model, the concurrent inference process of the model can be executed in Runtime. This tutorial describes how to perform concurrent inference with multiple modes by using the [Java interface](https://www.mindspore.cn/lite/api/en/master/api_java/class_list.html).

To use the MindSpore Lite parallel inference framework, perform the following steps:

1. Create a configuration item: Create a multi-model concurrent inference configuration item [RunnerConfig](https://www.mindspore.cn/lite/api/en/master/api_java/runner_config.html#runnerconfig), which is used to configure multiple model concurrency.
2. Initialization: initialization before multi-model concurrent inference.
3. Execute concurrent inference: Use the Predict interface of ModelParallelRunner to perform concurrent inference on multiple Models.
4. Release memory: When you do not need to use the MindSpore Lite concurrent inference framework, you need to release the ModelParallelRunner and related Tensors you created.

## Preparation

1. The following code samples are from [Sample code for performing cloud-side inference by C++ interface](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/quick_start_java).

2. Export the MindIR model via MindSpore, or get the MindIR model by converting it with [model conversion tool](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html) and copy it to the `mindspore/lite/examples/cloud_infer/quick_start_parallel_java/model` directory, and you can download the MobileNetV2 model file [mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir).

3. Download the Ascend, Nvidia GPU, CPU triplet MindSpore Lite cloud-side inference package `mindspore-lite-{version}-linux-{arch}.tar.gz` from [Official Website](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) and save it to `mindspore/lite/examples/cloud_infer/quick_start_parallel_java` directory.

## Creating Configuration

The [configuration item](https://www.mindspore.cn/lite/api/en/master/api_java/runner_config.html) will save some basic configuration parameters required for concurrent reasoning, which are used to guide the number of concurrent models, model compilation and model execution.

The following [sample code](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L83) demonstrates how to create a RunnerConfig and configure the number of workers for concurrent inference:

```java
// use default param init context
MSContext context = new MSContext();
context.init(1,0);
boolean ret = context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
if (!ret) {
    System.err.println("init context failed");
    context.free();
    return ;
}
// init runner config
RunnerConfig config = new RunnerConfig();
config.init(context);
config.setWorkersNum(2);
```

> For details on the configuration method of Context, see [Context](https://www.mindspore.cn/lite/docs/en/master/use/runtime_java.html#creating-a-configuration-context).
>
> Multi-model concurrent inference currently only supports [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/api_java/mscontext.html#devicetype), [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/master/api_java/mscontext.html#devicetype), and [AscendDeviceInfo](https://www.mindspore.cn/lite/api/en/master/api_java/mscontext.html#devicetype) several different hardware backends. When setting the GPU backend, you need to set the GPU backend first and then the CPU backend, otherwise it will report an error and exit.
>
> Multi-model concurrent inference does not support FP32 type data reasoning. Binding cores only supports no core binding or binding large cores. It does not support the parameter settings of the bound cores, and does not support configuring the binding core list.

## Initialization

When using MindSpore Lite to execute concurrent reasoning, ModelParallelRunner is the main entry of concurrent reasoning. Through ModelParallelRunner, you can initialize and execute concurrent reasoning. Use the RunnerConfig created in the previous step and call the init interface of ModelParallelRunner to initialize ModelParallelRunner.

The following [sample code](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L125) demonstrates the initialization process of ModelParallelRunner:

```java
ret = runner.predict(inputs,outputs);
if (!ret) {
    System.err.println("MindSpore Lite predict failed.");
    freeTensor();
    runner.free();
    return;
}
```

> For Initialization of ModelParallelRunner, you do not need to set the RunnerConfig configuration parameters, and the default parameters will be used for concurrent inference of multiple models.

## Execute concurrent inference

MindSpore Lite calls the Predict interface of ModelParallelRunner for model concurrent inference.

The following [main.cc](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L125) demonstrates how to call `Predict` to execute inference.

```java
ret = runner.predict(inputs,outputs);
if (!ret) {
    System.err.println("MindSpore Lite predict failed.");
    freeTensor();
    runner.free();
    return;
}
```

## Memory release

When you do not need to use the MindSpore Lite reasoning framework, you need to release the created ModelParallelRunner. The following [sample code](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L133) demonstrates how to free memory before the end of the program.

```java
freeTensor();
runner.free();
```
