# 使用Java接口执行并发推理

<a href="https://gitee.com/mindspore/docs/blob/r1.10/docs/lite/docs/source_zh_cn/use/runtime_server_inference_java.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png"></a>

## 概述

MindSpore Lite提供多model并发推理接口[ModelParallelRunner](https://www.mindspore.cn/lite/api/zh-CN/r1.10/api_java/model_parallel_runner.html)，多model并发推理现支持CPU、GPU后端。

> 快速了解MindSpore Lite执行并发推理的完整调用流程，请参考[体验Java极简并发推理Demo](https://www.mindspore.cn/lite/docs/zh-CN/r1.10/quick_start/quick_start_server_inference_java.html)。

通过[MindSpore Lite模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r1.10/use/converter_tool.html)转换成`.ms`模型后，即可在Runtime中执行模型的并发推理流程。本教程介绍如何使用[Java接口](https://www.mindspore.cn/lite/api/zh-CN/r1.10/api_java/class_list.html)执行多model并发推理。

使用MindSpore Lite并发推理主要包括以下步骤：

1. 创建配置项：创建多model并发推理配置项[RunnerConfig](https://www.mindspore.cn/lite/api/zh-CN/r1.10/api_java/runner_config.html)，用于配置多model并发。
2. 初始化：多model并发推理前的初始化。
3. 执行并发推理：使用ModelParallelRunner的Predict接口进行多Model并发推理。
4. 释放内存：无需使用MindSpore Lite并发推理框架时，需要释放自己创建的ModelParallelRunner以及相关的Tensor。

## 创建配置项

配置项[RunnerConfig](https://www.mindspore.cn/lite/api/zh-CN/r1.10/api_java/runner_config.html)会保存一些并发推理所需的基本配置参数，用于指导并发model数量以及模型编译和模型执行；

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.10/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L83)演示了如何创建RunnerConfig，并配置并发推理的worker数量。

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

> Context的配置方法详细见[Context](https://www.mindspore.cn/lite/docs/zh-CN/r1.10/use/runtime_java.html#创建配置上下文)。
>
> 多model并发推理现阶段只支持[CPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r1.10/api_java/mscontext.html#devicetype)、[GPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r1.10/api_java/mscontext.html#devicetype)两种不同的硬件后端。在设置GPU后端的时候需要先设置GPU后端再设置CPU后端，否则会报错退出。
>
> 多model并发推理不支持FP32类型数据推理，绑核只支持不绑核或者绑大核，不支持绑中核的参数设置，且不支持配置绑核列表。

## 初始化

使用MindSpore Lite执行并发推理时，ModelParallelRunner是并发推理的主入口，通过ModelParallelRunner可以初始化以及执行并发推理。采用上一步创建得到的RunnerConfig，调用ModelParallelRunner的Init接口来实现ModelParallelRunner的初始化。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.10/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L99)演示了ModelParallelRunner的初始化过程：

```java
// init ModelParallelRunner
ModelParallelRunner runner = new ModelParallelRunner();
ret = runner.init(modelPath, config);
if (!ret) {
    System.err.println("ModelParallelRunner init failed.");
    runner.free();
    return;
}
```

> ModelParallelRunner的初始化，可以不设置RunnerConfig配置参数，则会使用默认参数进行多model的并发推理。

## 执行并发推理

MindSpore Lite调用ModelParallelRunner的Predict接口进行模型并发推理。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.10/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L125)演示调用`Predict`执行推理。

```java
ret = runner.predict(inputs,outputs);
if (!ret) {
    System.err.println("MindSpore Lite predict failed.");
    freeTensor();
    runner.free();
    return;
}
```

## 释放内存

无需使用MindSpore Lite推理框架时，需要释放已经创建的ModelParallelRunner，下列[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.10/mindspore/lite/examples/quick_start_server_inference_java/src/main/java/com/mindspore/lite/demo/Main.java#L133)演示如何在程序结束前进行内存释放。

```java
freeTensor();
runner.free();
```
