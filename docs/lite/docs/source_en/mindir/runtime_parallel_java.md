# Using Java Interface to Perform Concurrent Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/mindir/runtime_parallel_java.md)

## Overview

MindSpore Lite provides multi-model concurrent inference interface [ModelParallelRunner](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model_parallel_runner.html#modelparallelrunner). Multi model concurrent inference now supports Atlas 200/300/500 inference product, Atlas inference series, Atlas training series, Nvidia GPU and CPU backends.

After exporting the `mindir` model by MindSpore or converting it by [model conversion tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool.html) to obtain the `mindir` model, the concurrent inference process of the model can be executed in Runtime. This tutorial describes how to perform concurrent inference with multiple modes by using the [Java interface](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/class_list.html).

To use the MindSpore Lite concurrent inference framework, perform the following steps:

1. Create a configuration item: Create a multi-model concurrent inference configuration item [RunnerConfig](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/runner_config.html#runnerconfig), which is used to configure multiple model concurrency.
2. Initialization: initialization before multi-model concurrent inference.
3. Execute concurrent inference: Use the Predict interface of ModelParallelRunner to perform concurrent inference on multiple Models.
4. Release memory: When you do not need to use the MindSpore Lite concurrent inference framework, you need to release the ModelParallelRunner and related Tensors you created.

![](./images/server_inference.png)

## Preparation

1. The following code samples are from [Sample code for performing cloud-side inference by C++ interface](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/quick_start_parallel_java).

2. Export the MindIR model via MindSpore, or get the MindIR model by converting it with [model conversion tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool.html) and copy it to the `mindspore/lite/examples/cloud_infer/quick_start_parallel_java/model` directory, and you can download the MobileNetV2 model file [mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir).

3. Download the Ascend, Nvidia GPU, CPU triplet MindSpore Lite cloud-side inference package `mindspore-lite-{version}-linux-{arch}.tar.gz` from [Official Website](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) and save it to `mindspore/lite/examples/cloud_infer/quick_start_parallel_java` directory.

## Creating Configuration

The [configuration item](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/runner_config.html) will save some basic configuration parameters required for concurrent inference, which are used to guide the number of concurrent models, model compilation and model execution.

The following sample code demonstrates how to create a RunnerConfig and configure the number of workers for concurrent inference:

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

> For details on the configuration method of Context, see [Context](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/runtime_java.html#creating-a-configuration-context).
>
> Multi-model concurrent inference currently only supports [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#devicetype), [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#devicetype), and [AscendDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#devicetype) several different hardware backends. When setting the GPU backend, you need to set the GPU backend first and then the CPU backend, otherwise it will report an error and exit.
>
> Multi-model concurrent inference does not support FP32 type data reasoning. Binding cores only supports no core binding or binding large cores. It does not support the parameter settings of the bound cores, and does not support configuring the binding core list.

## Initialization

When using MindSpore Lite to execute concurrent inference, ModelParallelRunner is the main entry of concurrent inference. Through ModelParallelRunner, you can initialize and execute concurrent inference. Use the RunnerConfig created in the previous step and call the init interface of ModelParallelRunner to initialize ModelParallelRunner.

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

## Executing Concurrent Inference

MindSpore Lite calls the Predict interface of ModelParallelRunner for model concurrent inference.

```java
ret = runner.predict(inputs,outputs);
if (!ret) {
    System.err.println("MindSpore Lite predict failed.");
    freeTensor();
    runner.free();
    return;
}
```

## Building and Running

### Build

Set environment variables, and Run the [build script](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/quick_start_parallel_java/build.sh) in the `mindspore/lite/examples/quick_start_parallel_java` directory to automatically download the MindSpore Lite inference framework library and model files and build the Demo.

```bash
export JAVA_HOME=/{path}/default-java
export M2_HOME=/{path}/maven
export MAVEN_HOME=/{path}/maven
export PATH=/{path}/maven/bin:$PATH

bash build.sh
```

### Inference

After the build, go to the `mindspore/lite/examples/cloud_infer/quick_start_parallel_java/target` directory and run the following command to experience MindSpore Lite inference on the MobileNetV2 model:

```java
java -classpath .:./quick_start_parallel_java.jar:../lib/runtime/lib/mindspore-lite-java.jar  com.mindspore.lite.demo.Main ../model/mobilenetv2.mindir
```

After the execution is completed, it will show the model concurrent inference success.

## Memory release

When you do not need to use the MindSpore Lite reasoning framework, you need to release the created ModelParallelRunner.

```java
freeTensor();
runner.free();
```
