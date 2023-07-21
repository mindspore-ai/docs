# 使用Runtime执行推理（Java）

`Android` `推理应用` `模型加载` `数据准备` `中级` `高级`

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/lite/source_zh_cn/use/runtime_java.md)

## 概述

通过MindSpore Lite模型转换后，需在Runtime中完成模型的推理执行流程。本教程介绍如何使用Java接口编写推理代码。

> 更多Java API说明，请参考 [API文档](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/index.html)。

## Android项目引用AAR包

首先将`mindspore-lite-{version}.aar`文件移动到目标module的**libs**目录，然后在目标module的`build.gradle`的`repositories`中添加本地引用目录，最后在`dependencies`中添加aar的依赖，具体如下所示。

```groovy
repositories {
    flatDir {
        dirs 'libs'
    }
}

dependencies {
    implementation(name:'mindspore-lite-{version}', ext:'aar')
}
```

> 注意mindspore-lite-{version}是aar的文件名，需要将{version}替换成对应版本信息。

## 运行MindSpore Lite推理框架

Android项目中使用MindSpore Lite，可以选择采用C++ APIs或者Java APIs运行推理框架。Java APIs与C++ APIs相比较而言，Java APIs可以直接在Java Class中调用，无需实现JNI层的相关代码，具有更好的便捷性。运行Mindspore Lite推理框架主要包括以下步骤：

1. 加载模型：从文件系统中读取MindSpore Lite模型，并进行模型解析。
2. 创建配置上下文：创建配置上下文[MSConfig](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/msconfig.html#msconfig)，保存会话所需的一些基本配置参数，用于指导图编译和图执行。主要包括`deviceType`：设备类型、`threadNum`：线程数、`cpuBindMode`：CPU绑定模式、`enable_float16`：是否优先使用float16算子。
3. 创建会话：创建[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)，并调用[init](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#init)方法将上一步得到[MSConfig](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/msconfig.html#msconfig)配置到会话中。
4. 图编译：在图执行前，需要调用[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)的[compileGraph](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#compilegraph)接口进行图编译，主要进行子图切分、算子选型调度。这部分会耗费较多时间，所以建议[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)创建一次，编译一次，多次执行。
5. 输入数据：图执行之前需要向输入Tensor中填充数据。
6. 图执行：使用[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)的[runGraph](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#rungraph)进行模型推理。
7. 获得输出：图执行结束之后，可以通过输出Tensor得到推理结果。
8. 释放内存：无需使用MindSpore Lite推理框架的时候，需要将创建的[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)和[model](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/model.html#model)进行释放。

### 加载模型

MindSpore Lite进行模型推理时，需要先从文件系统中加载模型转换工具转换后的`.ms`模型，并进行模型解析。Java的[model](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/model.html#model)类提供了[loadModel](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/model.html#loadmodel)，使其可以从`Assets`或其他文件路径中加载模型。

```java
// Load the .ms model.
model = new Model();
if (!model.loadModel(context, "model.ms")) {
    Log.e("MS_LITE", "Load Model failed");
    return false;
}
```

### 创建配置上下文

创建配置上下文[MSConfig](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/msconfig.html#msconfig)，保存会话所需的一些基本配置参数，用于指导图编译和图执行。

MindSpore Lite支持异构推理，推理时的主选后端由[MSConfig](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/msconfig.html#msconfig)的`deviceType`指定，目前支持CPU和GPU。在进行图编译时，会根据主选后端进行算子选型调度。

MindSpore Lite内置一个进程共享的线程池，推理时通过`threadNum`指定线程池的最大线程数，默认为2线程，推荐最多不超过4个线程，否则可能会影响性能。

MindSpore Lite支持float16算子的模式进行推理。`enable_float16`设置为`true`后，将会优先使用float16算子。

```java
// Create and init config.
MSConfig msConfig = new MSConfig();
if (!msConfig.init(DeviceType.DT_CPU, 2, CpuBindMode.MID_CPU, true)) {
    Log.e("MS_LITE", "Init context failed");
    return false;
}
```

### 创建会话

[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)是推理的主入口，通过[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)可以进行图编译、图执行。创建[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)，并调用[init](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#init)方法将上一步得到[MSConfig](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/msconfig.html#msconfig)配置到会话中。[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)初始化之后，[MSConfig](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/msconfig.html#msconfig)将可以进行释放操作。

```java
// Create the MindSpore lite session.
session = new LiteSession();
if (!session.init(msConfig)) {
    Log.e("MS_LITE", "Create session failed");
    msConfig.free();
    return false;
}
msConfig.free();
```

### 图编译

在图执行前，需要调用[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)的[compileGraph](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#compilegraph)接口进行图编译，主要进行子图切分、算子选型调度。这部分会耗费较多时间，所以建议[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)创建一次，编译一次，多次执行。图编译结束之后，可以调用[Model](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/model.html#model)的[freeBuffer](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/model.html#freebuffer)函数，释放MindSpore Lite Model中的MetaGraph，用于减小运行时的内存，但释放后该[Model](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/model.html#model)就不能再次图编译。

```java
// Compile graph.
if (!session.compileGraph(model)) {
    Log.e("MS_LITE", "Compile graph failed");
    model.freeBuffer();
    return false;
}

// Note: when use model.freeBuffer(), the model can not be compiled.
model.freeBuffer();
```

### 输入数据

Java目前支持`byte[]`或者`ByteBuffer`两种类型的数据，设置输入Tensor的数据。

```java
// Set input tensor values.
List<MSTensor> inputs = session.getInputs();
MSTensor inTensor = inputs.get(0);
byte[] inData = readFileFromAssets(context, "model_inputs.bin");
inTensor.setData(inData);
```

### 图执行

通过[LiteSession](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)的[runGraph](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#rungraph)执行模型推理。

```java
// Run graph to infer results.
if (!session.runGraph()) {
    Log.e("MS_LITE", "Run graph failed");
    return;
}
```

### 获得输出

推理结束之后，可以通过输出Tensor得到推理结果。目前输出tensor支持的数据类型包括`float`、`int`、`long`、`byte`。

- 获得输出Tensor的方法有三种：
    - 使用[getOutputMapByTensor](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#getoutputmapbytensor)方法，直接获取所有的模型输出[MSTensor](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/mstensor.html#mstensor)的名称和[MSTensor](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/mstensor.html#mstensor)指针的一个map。
    - 使用[GetOutputsByNodeName](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#getoutputsbynodename)方法，根据模型输出节点的名称来获取模型输出[MSTensor](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/mstensor.html#mstensor)中连接到该节点的Tensor的vector。
    - 使用[GetOutputByTensorName](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#getoutputbytensorname)方法，根据模型输出Tensor的名称来获取对应的模型输出[MSTensor](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/mstensor.html#mstensor)。

```java
// Get output tensor values.
List<String> tensorNames = session.getOutputTensorNames();
Map<String, MSTensor> outputs = session.getOutputMapByTensor();
Set<Map.Entry<String, MSTensor>> entries = outputs.entrySet();
for (String tensorName : tensorNames) {
    MSTensor output = outputs.get(tensorName);
    if (output == null) {
        Log.e("MS_LITE", "Can not find output " + tensorName);
        return;
    }
    float[] results = output.getFloatData();

    // Apply infer results.
    ……
}
```

### 释放内存

无需使用MindSpore Lite推理框架的时候，需要将创建的[session](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/lite_session.html#litesession)和[model](https://www.mindspore.cn/doc/api_java/zh-CN/r1.1/model.html#model)进行释放。

```java
private void free() {
    session.free();
    model.free();
}
```

## Android项目使用MindSpore Lite推理框架示例

采用MindSpore Lite Java API推理主要包括`加载模型`、`创建配置上下文`、`创建会话`、`图编译`、`输入数据`、`图执行`、`获得输出`、`释放内存`等步骤。

```java
private boolean init(Context context) {
    // Load the .ms model.
    model = new Model();
    if (!model.loadModel(context, "model.ms")) {
        Log.e("MS_LITE", "Load Model failed");
        return false;
    }

    // Create and init config.
    MSConfig msConfig = new MSConfig();
    if (!msConfig.init(DeviceType.DT_CPU, 2, CpuBindMode.MID_CPU)) {
        Log.e("MS_LITE", "Init context failed");
        return false;
    }

    // Create the MindSpore lite session.
    session = new LiteSession();
    if (!session.init(msConfig)) {
        Log.e("MS_LITE", "Create session failed");
        msConfig.free();
        return false;
    }
    msConfig.free();

    // Compile graph.
    if (!session.compileGraph(model)) {
        Log.e("MS_LITE", "Compile graph failed");
        model.freeBuffer();
        return false;
    }

    // Note: when use model.freeBuffer(), the model can not be compiled.
    model.freeBuffer();

    return true;
}

private void DoInference(Context context) {
    // Set input tensor values.
    List<MSTensor> inputs = session.getInputs();
    MSTensor inTensor = inputs.get(0);
    byte[] inData = readFileFromAssets(context, "model_inputs.bin");
    inTensor.setData(inData);

    // Run graph to infer results.
    if (!session.runGraph()) {
        Log.e("MS_LITE", "Run graph failed");
        return;
    }

    // Get output tensor values.
    List<String> tensorNames = session.getOutputTensorNames();
    Map<String, MSTensor> outputs = session.getOutputMapByTensor();
    Set<Map.Entry<String, MSTensor>> entries = outputs.entrySet();
    for (String tensorName : tensorNames) {
        MSTensor output = outputs.get(tensorName);
        if (output == null) {
            Log.e("MS_LITE", "Can not find output " + tensorName);
            return;
        }
        float[] results = output.getFloatData();

        // Apply infer results.
        ……
    }
}

// Note: we must release the memory at the end, otherwise it will cause the memory leak.
private void free() {
    session.free();
    model.free();
}
```
