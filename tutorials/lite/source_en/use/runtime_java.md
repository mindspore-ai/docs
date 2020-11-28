# Using Runtime for Model Inference (Java)

<!-- TOC -->

- [Using Runtime for Model Inference (Java)](#using-runtime-for-model-inference-java)
    - [Overview](#overview)
    - [Android project references AAR package](#android-project-references-aar-package)
    - [Running MindSpore Lite inference framework](#running-mindspore-lite-inference-framework)
        - [Loading Model](#loading-model)
        - [Creating Configuration Context](#creating-configuration-context)
        - [Creating Session](#creating-session)
        - [Compiling Graphs](#compiling-graphs)
        - [Setting Data](#setting-data)
        - [Graph Execution](#graph-execution)
        - [Getting Output](#getting-output)
        - [Releasing Memory](#releasing-memory)
    - [Example of Android project using MindSpore Lite inference framework](#example-of-android-project-using-mindspore-lite-inference-framework)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_en/use/runtime_java.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

After model conversion using MindSpore Lite, the model inference process needs to be completed in Runtime. This tutorial introduces how to use Java API to write inference code.

> For more details for Java API, please refer to [API Docs](https://www.mindspore.cn/doc/api_java/en/master/index.html).

## Android project references AAR package

First copy the `mindspore-lite-{version}.aar` file to the **libs** directory of the target module, then add the local reference directory to the `repositories` of the target module's `build.gradle`, and finally, add the AAR package in the `dependencies`. The details are as follows.

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

> mindspore-lite-{version} is the name of the AAR file, you need to replace {version} with the corresponding version information.

## Running MindSpore Lite inference framework

Using MindSpore Lite in the Android project, you can choose to use C++ APIs or Java APIs to run the inference framework. Compared with C++ APIs, Java APIs can be called directly in Java Class without the need to implement the relevant code of the JNI layer, which is more convenient. Running the MindSpore Lite inference framework mainly includes the following steps:

1. Loading model: Read the MindSpore Lite model from the file system and parse the model.
2. Creating configuration context:  `MSConfig` saves some basic configuration parameters required by the session, which is used to guide graph compilation and graph execution. `MSConfig` mainly include `deviceType`: [device type](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/app/src/main/java/com/mindspore/lite/config/DeviceType.java), `threadNum`: number of threads, `cpuBindMode`: [CPU binding mode](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/app/src/main/java/com/mindspore/lite/config/CpuBindMode.java), and `enable_float16`: whether to use float16 operator as priority.
3. Creating session: Create the `LiteSession` and call the `init` method to configure the `MSConfig` obtained in the previous step into the session.
4. Compiling graphs: Before graph execution, call the `compileGraph` API of the `LiteSession` to compile graphs, mainly for subgraph split and operator selection and scheduling. This process takes a long time. Therefore, it is recommended that `LiteSession` achieves multiple executions with one creation and one compilation.
5. Setting data: Before the graph is executed, data needs to be set in the input Tensor.
6. Graph execution: Run model inference using `runGraph` of `LiteSession`.
7. Getting output: After the execution of the graph is finished, the inference result can be obtained by output Tensor.
8. Releasing memory: When you finishing using the MindSpore Lite inference framework, you need to release the created `session` and `model`.

### Loading Model

When MindSpore Lite runs model inference, it is necessary to load the `.ms` model converted by the model conversion tool from the file system and perform model analysis.  `Model` class provides loadModel so that it can load models from `Assets` or other file paths.

```java
// Load the .ms model.
model = new Model();
if (!model.loadModel(context, "model.ms")) {
    Log.e("MS_LITE", "Load Model failed");
    return false;
}
```

### Creating Configuration Context

 `MSConfig` saves some basic configuration parameters required by the session, which is used to guide graph compilation and graph execution

MindSpore Lite supports heterogeneous inference. The preferred backend for inference is specified by `deviceType` in `MSConfig` and is CPU by default. During graph compilation, operator selection and scheduling are performed based on the preferred backend.

MindSpore Lite has a built-in thread pool shared by processes. During inference, `threadNum` is used to specify the maximum number of threads in the thread pool. The default maximum number is 2. It is recommended that the maximum number does not exceed 4. Otherwise, the performance may be affected.

MindSpore Lite supports the float16 operator mode for reasoning. If `enable float16` is set as `true`, the float16 operator will be used first.

```java
// Create and init config.
MSConfig msConfig = new MSConfig();
if (!msConfig.init(DeviceType.DT_CPU, 2, CpuBindMode.MID_CPU, true)) {
    Log.e("MS_LITE", "Init context failed");
    return false;
}
```

### Creating Session

`LiteSession` is the main entrance of inference, graph compilation and graph execution can be done through `LiteSession`. Create a `LiteSession` and call the `init` method to configure the `MSConfig` obtained in the previous step into the session. After `LiteSession` is initialized, `MSConfig` can be released.

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

### Compiling Graphs

Before graph execution, call the `compileGraph` API of the `LiteSession` to compile graphs and further parse the Model instance loaded from the file, mainly for subgraph split and operator selection and scheduling. This process takes a long time. Therefore, it is recommended that `LiteSession` achieves multiple executions with one creation and one compilation. After the graph is compiled, you can call the `freeBuffer()` function of the `Model` to release the MetaGraph in the MindSpore Lite Model, which is used to reduce the runtime memory, but the model cannot be compiled again after being released.

```java
// Complile graph.
if (!session.compileGraph(model)) {
    Log.e("MS_LITE", "Compile graph failed");
    model.freeBuffer();
    return false;
}

// Note: when use model.freeBuffer(), the model can not be compliled.
model.freeBuffer();
```

### Setting Data

Java currently supports two types of data: `byte[]` or `ByteBuffer`, and set the input Tensor data.

```java
// Set input tensor values.
List<MSTensor> inputs = session.getInputs();
MSTensor inTensor = inputs.get(0);
byte[] inData = readFileFromAssets(context, "model_inputs.bin");
inTensor.setData(inData);
```

### Graph Execution

Run model inference using `runGraph` of `LiteSession`.

```java
// Run graph to infer results.
if (!session.runGraph()) {
    Log.e("MS_LITE", "Run graph failed");
    return;
}
```

### Getting Output

After the inference is finished, the inference result can be obtained by output Tensor. The data types currently supported by the output tensor include `float`, `int`, `long`, and `byte`.

- There are three ways to obtain the output Tensor:
    - Use the `getOutputMapByTensor` method to directly obtain the mapping between the names of all model output tensors and the model output `MSTensor`.
    - Use the `getOutputsByNodeName` method to obtain vectors of the model output `MSTensor` that is connected to the model output node based on the node name.
    - Use the `getOutputByTensorName` method to obtain the model output `MSTensor` based on the tensor name.

```java
// Get output tensor values.
List<String> tensorNames = session.getOutputTensorNames();
Map<String, MSTensor> outputs = session.getOutputMapByTensor();
Set<Map.Entry<String, MSTensor>> entrys = outputs.entrySet();
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

### Releasing Memory

When you finish using the MindSpore Lite inference framework, you need to release the created `session` and `model`.

```java
private void free() {
    session.free();
    model.free();
}
```

## Example of Android project using MindSpore Lite inference framework

Reasoning using the MindSpore Lite Java API mainly includes the steps of `Loading Model`, `Create configuration context`, `Creating sessions`, `Compiling Graphs`, `Setting Data`, `Graph Execution`, `Getting Output`, and `Releasing Memory`.

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

    // Complile graph.
    if (!session.compileGraph(model)) {
        Log.e("MS_LITE", "Compile graph failed");
        model.freeBuffer();
        return false;
    }

    // Note: when use model.freeBuffer(), the model can not be compliled.
    model.freeBuffer();

    return true;
}

private void DoInference(Context context) {
    // Set input tensor values.
    List<MSTensor> inputs = session.getInputs();
    byte[] inData = readFileFromAssets(context, "model_inputs.bin");
    MSTensor inTensor = inputs.get(0);
    inTensor.setData(inData);

    // Run graph to infer results.
    if (!session.runGraph()) {
        Log.e("MS_LITE", "Run graph failed");
        return;
    }

    // Get output tensor values.
    List<String> tensorNames = session.getOutputTensorNames();
    Map<String, MSTensor> outputs = session.getOutputMapByTensor();
    Set<Map.Entry<String, MSTensor>> entrys = outputs.entrySet();
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
