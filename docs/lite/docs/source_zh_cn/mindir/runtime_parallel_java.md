# 使用Java接口执行并发推理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/mindir/runtime_parallel_java.md)

## 概述

MindSpore Lite提供多model并发推理接口[ModelParallelRunner](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model_parallel_runner.html)，多model并发推理现支持Atlas 200/300/500推理产品、Atlas推理系列产品、Atlas训练系列产品、Nvidia GPU、CPU后端。

通过MindSpore导出`mindir`模型，或者由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool.html)转换获得`mindir`模型后，即可在Runtime中执行模型的并发推理流程。本教程介绍如何使用[Java接口](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/class_list.html)执行多model并发推理。

使用MindSpore Lite并发推理主要包括以下步骤：

1. 创建配置项：创建多model并发推理配置项[RunnerConfig](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/runner_config.html)，用于配置多model并发。
2. 初始化：多model并发推理前的初始化。
3. 执行并发推理：使用ModelParallelRunner的Predict接口进行多Model并发推理。
4. 释放内存：无需使用MindSpore Lite并发推理框架时，需要释放自己创建的ModelParallelRunner以及相关的Tensor。

![](./images/server_inference.png)

## 准备工作

1. 以下代码样例来自于[使用Java接口执行云侧推理示例代码](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/quick_start_parallel_java)。

2. 通过MindSpore导出MindIR模型，或者由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool.html)转换获得MindIR模型，并将其拷贝到`mindspore/lite/examples/cloud_infer/quick_start_parallel_java/model`目录，可以下载MobileNetV2模型文件[mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir)。

3. 从[官网](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/use/downloads.html)下载Ascend、Nvidia GPU、CPU三合一的MindSpore Lite云侧推理包`mindspore-lite-{version}-linux-{arch}.tar.gz`，并存放到`mindspore/lite/examples/cloud_infer/quick_start_parallel_java`目录。

## 创建配置项

配置项[RunnerConfig](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/runner_config.html)会保存一些并发推理所需的基本配置参数，用于指导并发model数量以及模型编译和模型执行；

下面示例代码演示了如何创建RunnerConfig，并配置并发推理的worker数量。

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

> Context的配置方法详见[Context](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/runtime_java.html#创建配置上下文)。
>
> 多model并发推理现阶段支持[CPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html#devicetype)、[GPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html#devicetype)、[AscendDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html#devicetype)几种不同的硬件后端。在设置GPU后端的时候需要先设置GPU后端再设置CPU后端，否则会报错退出。
>
> 多model并发推理不支持FP32类型数据推理，绑核只支持不绑核或者绑大核，不支持绑中核的参数设置，且不支持配置绑核列表。

## 初始化

使用MindSpore Lite执行并发推理时，ModelParallelRunner是并发推理的主入口，通过ModelParallelRunner可以初始化以及执行并发推理。采用上一步创建得到的RunnerConfig，调用ModelParallelRunner的Init接口来实现ModelParallelRunner的初始化。

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

```java
ret = runner.predict(inputs,outputs);
if (!ret) {
    System.err.println("MindSpore Lite predict failed.");
    freeTensor();
    runner.free();
    return;
}
```

## 构建与运行

### 编译构建

设置环境变量，在`mindspore/lite/examples/cloud_infer/quick_start_parallel_java`目录下执行[build脚本](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/quick_start_parallel_java/build.sh)，将自动下载MindSpore Lite推理框架库以及模型文件并编译Demo。

```bash
export JAVA_HOME=/{path}/default-java
export M2_HOME=/{path}/maven
export MAVEN_HOME=/{path}/maven
export PATH=/{path}/maven/bin:$PATH

bash build.sh
```

### 执行推理

编译构建后，进入`mindspore/lite/examples/cloud_infer/quick_start_parallel_java/target`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

```java
java -classpath .:./quick_start_parallel_java.jar:../lib/runtime/lib/mindspore-lite-java.jar  com.mindspore.lite.demo.Main ../model/mobilenetv2.mindir
```

执行完成后显示模型并发推理成功。

## 释放内存

无需使用MindSpore Lite推理框架时，需要释放已经创建的ModelParallelRunner。

```java
freeTensor();
runner.free();
```
