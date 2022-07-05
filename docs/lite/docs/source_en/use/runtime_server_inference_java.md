# Using Java Interface to Parallel Inference

<a href="https://gitee.com/mindspore/docs/blob/r1.8/docs/lite/docs/source_en/use/runtime_server_inference_java.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore Lite provides multi-model concurrent inference interface [ModelParallelRunner](https://www.mindspore.cn/lite/api/en/r1.8/api_java/model_parallel_runner.html#modelparallelrunner). Multi model concurrent inference now supports CPU and GPU backend.

> For a quick understanding of the complete calling process of MindSpore Lite executing concurrent reasoning, please refer to [Experience Java Minimalist Concurrent Reasoning Demo](https://www.mindspore.cn/lite/docs/en/r1.8/quick_start/quick_start_server_inference_java.html).

After the model is converted into a `.ms` model by using the MindSpore Lite model conversion tool, the inference process can be performed in Runtime. For details, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/r1.8/use/converter_tool.html). This tutorial describes how to use the [Java API](https://www.mindspore.cn/lite/api/en/r1.8/index.html) to perform inference.

To use the MindSpore Lite parallel inference framework, perform the following steps:

1. Create a configuration item: Create a multi-model concurrent inference configuration item [RunnerConfig](https://www.mindspore.cn/lite/api/en/r1.8/api_java/runner_config.html#runnerconfig), which is used to configure multiple model concurrency.
2. Initialization: initialization before multi-model concurrent inference.
3. Execute concurrent inference: Use the Predict interface of ModelParallelRunner to perform concurrent inference on multiple Models.
4. Release memory: When you do not need to use the MindSpore Lite concurrent inference framework, you need to release the ModelParallelRunner and related Tensors you created.

## Create configuration

The [configuration item](https://www.mindspore.cn/lite/api/en/r1.8/api_java/runner_config.html) will save some basic configuration parameters required for concurrent reasoning, which are used to guide the number of concurrent models, model compilation and model execution.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/r1.8/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L83) demonstrates how to create a RunnerConfig and configure the number of workers for concurrent inference:

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

> For details on the configuration method of Context, see [Context](https://www.mindspore.cn/lite/docs/en/r1.8/use/runtime_java.html#creating-a-configuration-context).
>
> Multi-model concurrent inference currently only supports [CPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r1.8/api_java/mscontext.html#devicetype) and [GPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r1.8/api_java/mscontext.html#devicetype) two different hardware backends. When setting the GPU backend, you need to set the GPU backend first and then the CPU backend, otherwise it will report an error and exit.
>
> Multi-model concurrent inference does not support FP32 type data reasoning. Binding cores only supports no core binding or binding large cores. It does not support the parameter settings of the bound cores, and does not support configuring the binding core list.

## Initialization

When using MindSpore Lite to execute concurrent reasoning, ModelParallelRunner is the main entry of concurrent reasoning. Through ModelParallelRunner, you can initialize and execute concurrent reasoning. Use the RunnerConfig created in the previous step and call the init interface of ModelParallelRunner to initialize ModelParallelRunner.

The following sample code from [main.cc](https://gitee.com/mindspore/mindspore/blob/r1.8/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L125) demonstrates how to call Predict to execute inference:

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

The following [main.cc](https://gitee.com/mindspore/mindspore/blob/r1.8/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L125) demonstrates how to call `Predict` to execute inference.

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

When you do not need to use the MindSpore Lite reasoning framework, you need to release the created ModelParallelRunner. The following [main.cc](https://gitee.com/mindspore/mindspore/blob/r1.8/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L133) demonstrates how to free memory before the end of the program.

```java
freeTensor();
runner.free();
```
