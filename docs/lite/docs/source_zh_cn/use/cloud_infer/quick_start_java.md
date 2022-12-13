# 体验Java极简推理Demo - 云侧

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/use/cloud_infer/quick_start_java.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

本教程提供了MindSpore Lite执行云侧推理的示例程序，通过随机输入、执行推理、打印推理结果的方式，演示了利用[MindSpore Lite Java API](https://www.mindspore.cn/lite/api/zh-CN/master/index.html)进行推理的基本流程，用户能够快速了解MindSpore Lite执行推理相关Java API的使用。本教程通过随机生成的数据作为输入数据，执行MobileNetV2模型的推理，打印获得输出数据。相关代码请参考[mindspore/lite/examples/cloud_infer/quick_start_java](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/quick_start_java)目录。

Lite云侧推理仅支持在Linux环境部署运行。支持Ascend310、Ascend310P、Nvidia GPU和CPU硬件后端。本教程以CPU后端为例，如需使用Ascend、GPU后端以及其他高级用法，请参考[使用Runtime执行推理（Java）](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/runtime_java.html)。

如需体验Lite端侧推理流程，请参考文档[体验Java极简推理Demo-端侧](https://www.mindspore.cn/lite/docs/zh-CN/master/quick_start/quick_start_java.html)。

使用MindSpore Lite执行推理主要包括以下步骤：

1. 模型读取：从文件系统中读取由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)转换得到的`.mindir`模型。
2. 创建配置上下文：创建配置上下文[MSContext](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/mscontext.html#mscontext)，保存会话所需的一些基本配置参数，用于指导图编译和图执行。
3. 模型创建、加载与编译：执行推理之前，需要调用[Model](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/model.html#model)的[build](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/model.html#build)接口进行进行模型加载和模型编译，并将上一步得到的Context配置到Model中。模型加载阶段将文件缓存解析成运行时的模型。
4. 输入数据：图执行之前需要向输入Tensor中填充数据。
5. 执行推理：使用[model](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/model.html#model)的[predict](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/model.html#predict)方法进行模型推理。
6. 获得输出：图执行结束之后，可以通过输出Tensor得到推理结果。
7. 释放内存：无需使用MindSpore Lite推理框架的时候，需要释放已创建的[model](https://www.mindspore.cn/lite/api/zh-CN/master/api_java/model.html#model)。

![img](../../images/lite_runtime.png)

## 构建与运行

- 环境要求
    - 系统环境：Linux x86_64或arm64，推荐使用Ubuntu 18.04.02LTS。
    - 编译依赖：
        - [Git](https://git-scm.com/downloads) >= 2.28.0
        - [Maven](https://maven.apache.org/download.cgi) >= 3.3
        - [OpenJDK](https://openjdk.java.net/install/) 1.8 到 1.15
    > Ascend、GPU后端环境配置参考编译云侧MindSpore Lite中[环境准备](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/build.html#环境准备)章节。

- 编译构建

  在`mindspore/lite/examples/cloud_infer/quick_start_java`目录下执行[build脚本](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_java/build.sh)，将自动下载MindSpore Lite推理框架库以及文模型文件并编译Demo。

  ```bash
  source build.sh
  ```

  > 若使用该build脚本下载MindSpore Lite推理框架失败，请手动下载Ascend、Nvidia GPU、CPU三合一版本包。
  >
  > 若操作系统为Linux-x64，下载MindSpore Lite模型推理框架[mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)；若操作系统为Linux arm64，下载arm版本[mindspore-lite-{version}-linux-aarch64.tar.gz](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)。
  >
  > 将下载好的MindSpore Lite模型推理框架存放到`mindspore/lite/examples/cloud_infer/quick_start_java`目录。
  >
  > 若MobileNetV2模型下载失败，请手动下载相关模型文件[mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir)，并将其拷贝到`mindspore/lite/examples/cloud_infer/quick_start/model`目录。
  >
  > 通过手动下载并且将文件放到指定位置后，再次执行build.sh脚本完成编译构建。

- 执行推理
  > 注意：上一步中的`source build.sh`已包含`LD_LIBRARY_PATH`的添加，无需再手动执行。

  执行依赖MindSpore Lite包中如下目录的动态库，需添加至`LD_LIBRARY_PATH`。

  ```bash
  mindspore-lite-{version}-linux-x64/runtime/lib
  mindspore-lite-{version}-linux-x64/tools/converter/lib
  ```

  例如`$LITE_HOME`为包所在位置，添加`LD_LIBRARY_PATH`如下：

  ```bash
  export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
  ```

  编译构建后，进入`mindspore/lite/examples/quick_start_java/target`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  java -classpath .:./quick_start_java.jar:../lib/runtime/lib/mindspore-lite-java.jar  com.mindspore.lite.demo.Main ../model/mobilenetv2.mindir
  ```

  执行完成后将能得到如下结果，打印输出Tensor的大小和前50个数据：

  ```text
  out tensor shape: [1,1000,] and out data: 3.31223e-05 1.99382e-05 3.01624e-05 0.000108345 1.19685e-05 4.25282e-06 0.00049955 0.000340809 0.00199094 0.000997094 0.00013585 1.57605e-05 4.34131e-05 1.56114e-05 0.000550819 2.9839e-05 4.70447e-06 6.91601e-06 0.000134483 2.06795e-06 4.11612e-05 2.4667e-05 7.26248e-06 2.37974e-05 0.000134513 0.00142482 0.00011707 0.000161848 0.000395011 3.01961e-05 3.95325e-05 3.12398e-06 3.57709e-05 1.36277e-06 1.01068e-05 0.000350805 5.09019e-05 0.000805241 6.60321e-05 2.13734e-05 9.88654e-05 2.1991e-06 3.24065e-05 3.9479e-05 4.45178e-05 0.00205024 0.000780899 2.0633e-05 1.89997e-05 0.00197261 0.000259391
  ```

## 模型创建加载与编译

模型编译主要包括创建配置上下文、编译等步骤。下面描述的是从文件读取模型进行编译，需指定后端，以Ascend后端为例。

```java
private static boolean compile(String modelPath) {
    MSContext context = new MSContext();
    // use default param init context
    context.init();
    // Ascend: DT_ASCEND, CPU: DT_CPU, GPU: DT_GPU
    boolean ret = context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
    if (!ret) {
        System.err.println("Compile graph failed");
        context.free();
        return false;
    }
    // Create the MindSpore lite session.
    model = new Model();
    // Compile graph.
    ret = model.build(modelPath, ModelType.MT_MINDIR, context);
    if (!ret) {
        System.err.println("Compile graph failed");
        model.free();
        return false;
    }
    return true;
}
```

## 模型推理

模型推理主要包括输入数据、执行推理、获得输出等步骤。本示例中的输入数据是通过随机数据构造生成，最后将执行推理后的输出结果打印出来。

```java
private static boolean run() {
    MSTensor inputTensor = model.getInputs().get(0);
    if (inputTensor.getDataType() != DataType.kNumberTypeFloat32) {
        System.err.println("Input tensor data type is not float, the data type is " + inputTensor.getDataType());
        return false;
    }
    // Generator Random Data.
    int elementNums = inputTensor.elementsNum();
    float[] randomData = generateArray(elementNums);
    ByteBuffer inputData = floatArrayToByteBuffer(randomData);

    // Set Input Data.
    inputTensor.setData(inputData);

    // Run Inference.
    boolean ret = model.predict();
    if (!ret) {
        inputTensor.free();
        System.err.println("MindSpore Lite run failed.");
        return false;
    }

    // Get Output Tensor Data.
    MSTensor outTensor = model.getOutputs().get(0);

    // Print out Tensor Data.
    StringBuilder msgSb = new StringBuilder();
    msgSb.append("out tensor shape: [");
    int[] shape = outTensor.getShape();
    for (int dim : shape) {
        msgSb.append(dim).append(",");
    }
    msgSb.append("]");
    if (outTensor.getDataType() != DataType.kNumberTypeFloat32) {
        inputTensor.free();
        outTensor.free();
        System.err.println("output tensor data type is not float, the data type is " + outTensor.getDataType());
        return false;
    }
    float[] result = outTensor.getFloatData();
    if (result == null) {
        inputTensor.free();
        outTensor.free();
        System.err.println("decodeBytes return null");
        return false;
    }
    msgSb.append(" and out data:");
    for (int i = 0; i < 50 && i < outTensor.elementsNum(); i++) {
        msgSb.append(" ").append(result[i]);
    }
    System.out.println(msgSb.toString());
    // In/Out Tensor must be free
    inputTensor.free();
    outTensor.free();
    return true;
}
```

## 内存释放

无需使用MindSpore Lite推理框架时，需要释放已经创建的`model`。

```java
// Delete model buffer.
model.free();
```
