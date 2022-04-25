# 体验Java极简并发推理Demo

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/quick_start/quick_start_server_inference_java.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

本教程提供了MindSpore Lite执行并发推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了利用[MindSpore Lite Java API](https://www.mindspore.cn/lite/api/zh-CN/master/index.html)进行端侧并发推理的基本流程，用户能够快速了解MindSpore Lite执行并发推理相关Java API的使用。本教程通过随机生成的数据作为输入数据，执行MobileNetV2模型的推理，打印获得输出数据。相关代码放置在[mindspore/lite/examples/quick_start_server_inference_java](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/quick_start_server_inference_java)目录。

使用MindSpore Lite 推理主要包括以下步骤：

1. 模型加载：从文件系统中读取由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)转换得到的`.ms`模型。
2. 创建配置选项：创建配置选项[RunnerConfig](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/runner_config.html#runnerconfig)，保存并发推理所需的一些基本配置参数，用于指导并发量以及图编译和图执行。主要包括[MSContext](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/mscontext.html#mscontext)：配置上下文、`WorkersNum`：并发模型数量。
3. 初始化：在执行并发推理前，需要调用[ModelParallelRunner](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/model_parallel_runner.html#modelparallelrunner)的[init](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/model_parallel_runner.html#init)接口进行并发推理的初始化，主要进行模型读取，创建并发，以及子图切分、算子选型调度。这部分会耗费较多时间，所以建议[init](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/model_parallel_runner.html#init)初始化一次，多次执行并发推理。
4. 输入数据：图执行之前需要向输入Tensor中填充数据。
5. 执行推理：使用[ModelParallelRunner](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/model_parallel_runner.html#modelparallelrunner)的[predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/model_parallel_runner.html#predict)进行并发推理。
6. 获得输出：图执行结束之后，可以通过输出Tensor得到推理结果。
7. 释放内存：无需使用MindSpore Lite推理框架的时候，需要释放已创建的[ModelParallelRunner](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/model_parallel_runner.html#modelparallelrunner)。

![img](../images/server_inference.png)

## 构建与运行

- 环境要求
    - 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
    - 编译依赖：
        - [Git](https://git-scm.com/downloads) >= 2.28.0
        - [Maven](https://maven.apache.org/download.cgi) >= 3.3
        - [OpenJDK](https://openjdk.java.net/install/) >= 1.8

- 编译构建

  在`mindspore/lite/examples/quick_start_server_inference_java`目录下执行[build脚本](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_server_inference_java/build.sh)，将自动下载MindSpore Lite推理框架库以及文模型文件并编译Demo。

  ```bash
  bash build.sh
  ```

  > 若MindSpore Lite推理框架下载失败，请手动下载硬件平台为CPU、操作系统为Ubuntu-x64的MindSpore Lite 框架[mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)，解压后将`runtime/lib/mindspore-lite-java.jar`文件拷贝到`mindspore/lite/examples/quick_start_server_inference_java/lib`目录。
  >
  > 若MobileNetV2模型下载失败，请手动下载相关模型文件[mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms)，并将其拷贝到`mindspore/lite/examples/quick_start_server_inference_java/model/`目录。
  >
  > 通过手动下载并且将文件放到指定位置后，需要再次执行build.sh脚本才能完成编译构建。

- 执行推理

  编译构建后，进入`mindspore/lite/examples/quick_start_server_inference_java/target`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  java -classpath .:./quick_start_server_inference_java.jar:../lib/mindspore-lite-java.jar  com.mindspore.lite.demo.Main ../model/mobilenetv2.ms
  ```

  执行完成后将能得到如下结果:

  ```text
  ========== model parallel runner predict success ==========
  ```

## 初始化

模型编译主要包括创建配置上下文、编译等步骤。

```java
private static ModelParallelRunner runner;
private static List<MSTensor> inputs;
private static List<MSTensor> outputs;

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

// init ModelParallelRunner
ModelParallelRunner runner = new ModelParallelRunner();
ret = runner.init(modelPath, config);
if (!ret) {
    System.err.println("ModelParallelRunner init failed.");
    runner.free();
    return;
}
```

## 模型推理

模型推理主要包括输入数据、执行推理、获得输出等步骤，其中本示例中的输入数据是通过随机数据构造生成，最后将执行推理后的输出结果打印出来。

```java
// init input tensor
inputs = new ArrayList<>();
MSTensor input = runner.getInputs().get(0);
if (input.getDataType() != DataType.kNumberTypeFloat32) {
    System.err.println("Input tensor data type is not float, the data type is " + input.getDataType());
    return;
}
// Generator Random Data.
int elementNums = input.elementsNum();
float[] randomData = generateArray(elementNums);
ByteBuffer inputData = floatArrayToByteBuffer(randomData);
// create input MSTensor
MSTensor inputTensor = MSTensor.createTensor(input.tensorName(), DataType.kNumberTypeFloat32,input.getShape(), inputData);
inputs.add(inputTensor);

// init output
outputs = new ArrayList<>();

// runner do predict
ret = runner.predict(inputs,outputs);
if (!ret) {
    System.err.println("MindSpore Lite predict failed.");
    freeTensor();
    runner.free();
    return;
}
System.out.println("========== model parallel runner predict success ==========");
```

## 内存释放

无需使用MindSpore Lite推理框架时，需要释放已经创建的`model`。

```java
private static void freeTensor(){
    for (int i = 0; i < inputs.size(); i++) {
        inputs.get(i).free();
    }
    for (int i = 0; i < outputs.size(); i++) {
        outputs.get(i).free();
    }
}
freeTensor();
runner.free();
```
