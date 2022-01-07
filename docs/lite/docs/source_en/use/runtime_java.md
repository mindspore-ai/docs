# Using Java Interface to Perform Inference

`Android` `Java` `Inference Application` `Model Loading` `Data Preparation` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/runtime_java.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

After the model is converted into a `.ms` model by using the MindSpore Lite model conversion tool, the inference process can be performed in Runtime. For details, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html). This tutorial describes how to use the [Java API](https://www.mindspore.cn/lite/api/en/master/index.html) to perform inference.

If MindSpore Lite is used in an Android project, you can use [C++ API](https://www.mindspore.cn/lite/api/en/master/index.html) or [Java API](https://www.mindspore.cn/lite/api/en/master/index.html) to run the inference framework. Compared with C++ APIs, Java APIs can be directly called in the Java class. Users do not need to implement the code at the JNI layer, which is more convenient. To run the MindSpore Lite inference framework, perform the following steps:

1. Load the model: Read the `.ms` model converted by the model conversion tool introduced in [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html) from the file system and import the model using the [loadModel](https://www.mindspore.cn/lite/api/en/master/api_java/model.html#loadmodel).
2. Create a configuration context: Create a configuration context [MSConfig](https://www.mindspore.cn/lite/api/en/master/api_java/msconfig.html#msconfig) to save some basic configuration parameters required by a session to guide graph build and execution, including `deviceType` (device type), `threadNum` (number of threads), `cpuBindMode` (CPU core binding mode), and `enable_float16` (whether to preferentially use the float16 operator).
3. Create a session: Create [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) and call the [init](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#init) method to configure the [MSConfig](https://www.mindspore.cn/lite/api/en/master/api_java/msconfig.html#msconfig) obtained in the previous step in the session.
4. Build a graph: Before building a graph, the [compileGraph](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#compilegraph) API of [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) needs to be called to build the graph, including graph partition and operator selection and scheduling. This takes a long time. Therefore, it is recommended that with [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) created each time, one graph be built. In this case, the inference will be performed for multiple times.
5. Input data: Before the graph is performed, data needs to be filled in to the `Input Tensor`.
6. Perform inference: Use the [runGraph](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#rungraph) of the [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) to perform model inference.
7. Obtain the output: After the graph execution is complete, you can obtain the inference result by `outputting the tensor`.
8. Release the memory: If the MindSpore Lite inference framework is not required, release the created [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) and [Model](https://www.mindspore.cn/lite/api/en/master/api_java/model.html#model).

![img](../images/lite_runtime.png)

> For details about the calling process of MindSpore Lite inference, see [Experience Java Simple Inference Demo](https://www.mindspore.cn/lite/docs/en/master/quick_start/quick_start_cpp.html).

## Referencing the MindSpore Lite Java Library

### Linux X86 Project Referencing the JAR Library

When using `Maven` as the build tool, you can copy `mindspore-lite-java.jar` to the `lib` directory in the root directory and add the dependency of the JAR package to `pom.xml`.

```xml
<dependencies>
    <dependency>
        <groupId>com.mindspore.lite</groupId>
        <artifactId>mindspore-lite-java</artifactId>
        <version>1.0</version>
        <scope>system</scope>
        <systemPath>${project.basedir}/lib/mindspore-lite-java.jar</systemPath>
    </dependency>
</dependencies>
```

> Add the paths of `libmindspore-lite.so` and `libminspore-lite-jni.so` to `java.library.path`.

### Android Projects Referencing the AAR Library

When `Gradle` is used as the build tool, move the `mindspore-lite-{version}.aar` file to the `libs` directory of the target module, and then add the local reference directory to `repositories` of `build.gradle` of the target module, add the AAR dependency to `dependencies` as follows:

> Note that mindspore-lite-{version} is the AAR file name. Replace {version} with the corresponding version information.

```groovy
repositories {
    flatDir {
        dirs 'libs'
    }
}

dependencies {
    implementation fileTree(dir: "libs", include: ['*.aar'])
}
```

## Loading a Model

Before performing model inference, MindSpore Lite needs to load the `.ms` model converted by the model conversion tool from the file system and parse the model. The [Model](https://www.mindspore.cn/lite/api/en/master/api_java/model.html#model) class of Java provides two [loadModel](https://www.mindspore.cn/lite/api/en/master/api_java/model.html#loadmodel) APIs to load models from `Assets` or other file paths.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L217) reads the `mobilenetv2.ms` model file from `Assets` to load the model.

```java
// Load the .ms model.
Model model = new Model();
String modelPath = "mobilenetv2.ms";
boolean ret = model.loadModel(this.getApplicationContext(), modelPath);
```

> Only the `AAR` library supports the API for loading model files from `Assert`.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/quick_start_java/src/main/java/com/mindspore/lite/demo/Main.java#L128) reads the model file from the `modelPath` path to load the model.

```java
Model model = new Model();
boolean ret = model.loadModel(modelPath);
```

## Creating a Configuration Context

Create the configuration context [MSConfig](https://www.mindspore.cn/lite/api/en/master/api_java/msconfig.html#msconfig) to save some basic configuration parameters required by the session to guide graph build and execution.

MindSpore Lite supports heterogeneous inference. The preferred backend for inference is specified by `deviceType` of [MSConfig](https://www.mindspore.cn/lite/api/en/master/api_java/msconfig.html#msconfig). Currently, CPU and GPU are supported. During graph build, operator selection and scheduling are performed based on the preferred backend.

MindSpore Lite has a built-in thread pool shared by processes. During inference, `threadNum` is used to specify the maximum number of threads in the thread pool. The default value is 2.

MindSpore Lite supports inference in float16 operator mode. After `enable_float16` is set to `true`, the float16 operator is preferentially used.

### Configuring the CPU Backend

If the backend to be performed is a CPU, you need to configure `DeviceType.DT_CPU` in [init](https://www.mindspore.cn/lite/api/en/master/api_java/msconfig.html#init) after `MSConfig` is created. In addition, the CPU supports the setting of the core binding mode and whether to preferentially use the float16 operator.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L59) demonstrates how to create a CPU backend, set the CPU core binding mode to large-core priority, and enable float16 inference:

```java
MSConfig msConfig = new MSConfig();
boolean ret = msConfig.init(DeviceType.DT_CPU, 2, CpuBindMode.HIGHER_CPU, true);
```

> Float16 takes effect only when the CPU is of the ARM v8.2 architecture. Other models and x86 platforms that are not supported are automatically rolled back to float32.

### Configuring the GPU Backend

If the backend to be performed is heterogeneous inference based on CPU and GPU, you need to configure `DeviceType.DT_GPU` in [init](https://www.mindspore.cn/lite/api/en/master/api_java/msconfig.html#init) after `MSConfig` is created. After the configuration, GPU-based inference is preferentially used. In addition, if enable_float16 is set to true, both the GPU and CPU preferentially use the float16 operator.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L69) demonstrates how to create the CPU and GPU heterogeneous inference backend and how to enable float16 inference for the GPU.

```java
MSConfig msConfig = new MSConfig();
boolean ret = msConfig.init(DeviceType.DT_GPU, 2, CpuBindMode.MID_CPU, true);
```

> Currently, the GPU can run only on Android mobile devices. Therefore, only the `AAR` library can be run.

## Creating a Session

[LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) is the main entry for inference. You can use [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) to build and perform graphs. Create [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) and call the [init](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#init) method to configure the [MSConfig](https://www.mindspore.cn/lite/api/en/master/api_java/msconfig.html#msconfig) obtained in the previous step in the session. After the [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) is initialized, the [MSConfig](https://www.mindspore.cn/lite/api/en/master/api_java/msconfig.html#msconfig) can perform the release operation.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L86) demonstrates how to create a `LiteSession`:

```java
LiteSession session = new LiteSession();
boolean ret = session.init(msConfig);
msConfig.free();
```

## Building a Graph

Before building a graph, the [compileGraph](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#compilegraph) API of [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) needs to be called to build the graph, including graph partition and operator selection and scheduling. This takes a long time. Therefore, it is recommended that with the [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) created each time, one graph be built. In this case, the inference will be performed for multiple times.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L87) demonstrates how to call `CompileGraph` to build a graph.

```java
boolean ret = session.compileGraph(model);
```

## Inputting Data

MindSpore Lite Java APIs provide the `getInputsByTensorName` and `getInputs` methods to obtain the input tensor. Both the `byte[]` and `ByteBuffer` data types are supported. You can set the data of the input tensor by calling [setData](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html#setdata).

1. Use the [getInputsByTensorName](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#getinputsbytensorname) method to obtain the tensor connected to the input node from the model input tensor based on the name of the model input tensor. The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L151) demonstrates how to call the `getInputsByTensorName` function to obtain the input tensor and fill in data.

    ```java
    MSTensor inputTensor = session.getInputsByTensorName("2031_2030_1_construct_wrapper:x");
    // Set Input Data.
    inputTensor.setData(inputData);
    ```

2. Use the [getInputs](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#getinputs) method to directly obtain the vectors of all model input tensors. The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L113) demonstrates how to call `getInputs` to obtain the input tensors and fill in the data.

    ```java
    List<MSTensor> inputs = session.getInputs();
    MSTensor inputTensor = inputs.get(0);
    // Set Input Data.
    inputTensor.setData(inputData);
    ```

> The data layout in the input tensor of the MindSpore Lite model must be `NHWC`. For more information about data pre-processing, see [Implementing an Image Segmentation Application](https://www.mindspore.cn/lite/docs/en/master/quick_start/image_segmentation.html#id10).

## Executing Inference

After a MindSpore Lite session builds a graph, it can call the [runGraph](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#rungraph) function of [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) to perform model inference.

The following sample code demonstrates how to call `runGraph` to perform inference.

```java
// Run graph to infer results.
boolean ret = session.runGraph();
```

## Obtaining the Output

After performing inference, MindSpore Lite can output a tensor to obtain the inference result. MindSpore Lite provides three methods to obtain the output [MSTensor](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html) of a model and supports the [getByteData](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html#getbytedata), [getFloatData](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html#getfloatdata), [getIntData](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html#getintdata) and [getLongData](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html#getlongdata) methods to obtain the output data.

1. Use the [getOutputMapByTensor](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#getoutputmapbytensor) method to directly obtain the names of all model output [MSTensor](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html#mstensor) and a map of the [MSTensor](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html#mstensor) pointer. The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L191) demonstrates how to call `getOutputMapByTensor` to obtain the output tensor.

    ```java
    Map<String, MSTensor> outTensors = session.getOutputMapByTensor();

    Iterator<Map.Entry<String, MSTensor>> entries = outTensors.entrySet().iterator();
    while (entries.hasNext()) {
        Map.Entry<String, MSTensor> entry = entries.next();
        // Apply infer results.
        ...
    }
    ```

2. Use the [getOutputByNodeName](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#getoutputsbynodename) method to obtain the vector of the tensor connected to the model output [MSTensor](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html#mstensor) based on the name of the model output node. The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L175) demonstrates how to call `getOutputByTensorName` to obtain the output tensor.

    ```java
    MSTensor outTensor = session.getOutputsByNodeName("Default/head-MobileNetV2Head/Softmax-op204");
    // Apply infer results.
    ...
    ```

3. Use the [getOutputByTensorName](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#getoutputbytensorname) method to obtain the model output [MSTensor](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html#mstensor) based on the name of the model output tensor. The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L182) demonstrates how to call `getOutputByTensorName` to obtain the output tensor.

    ```java
    MSTensor outTensor = session.getOutputByTensorName("Default/head-MobileNetV2Head/Softmax-op204");
    // Apply infer results.
    ...
    ```

## Releasing the Memory

If the MindSpore Lite inference framework is not required, you need to release the created LiteSession and Model. The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L204) demonstrates how to release the memory before the program ends.

```java
session.free();
model.free();
```

## Advanced Usage

### Optimizing the Memory Size

If there is a large limit on the running memory, call the [freeBuffer](https://www.mindspore.cn/lite/api/en/master/api_java/model.html#freebuffer) function of [Model](https://www.mindspore.cn/lite/api/en/master/api_java/model.html#model) after the graph build is complete to release the MetaGraph in the MindSpore Lite Model to reduce the running memory. Once the [freeBuffer](https://www.mindspore.cn/lite/api/en/master/api_java/model.html#freebuffer) of a [Model](https://www.mindspore.cn/lite/api/en/master/api_java/model.html#model) is called, the [Model](https://www.mindspore.cn/lite/api/en/master/api_java/model.html#model) cannot be built again.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L241) demonstrates how to call the `freeBuffer` interface of `Model` to release `MetaGraph` to reduce the memory size during running.

```java
// Compile graph.
ret = session.compileGraph(model);
...
// Note: when use model.freeBuffer(), the model can not be compiled.
model.freeBuffer();
```

### Core Binding Operations

The built-in thread pool of MindSpore Lite supports core binding and unbinding. By calling the [BindThread](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#bindthread) API, you can bind working threads in the thread pool to specified CPU cores for performance analysis. The core binding operation is related to the context specified by the user when the [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html) is created. The core binding operation sets the affinity between the thread and the CPU based on the core binding policy in the context.

Note that core binding is an affinity operation and may not be bound to a specified CPU core. It may be affected by system scheduling. In addition, after the core binding, you need to perform the unbinding operation after the code is performed.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L164) demonstrates how to bind to cores with the highest frequency first when performing inference.

```java
boolean ret = msConfig.init(DeviceType.DT_CPU, 2, CpuBindMode.HIGHER_CPU, true);
...
session.bindThread(true);
// Run Inference.
ret = session.runGraph();
session.bindThread(false);
```

> There are three options for core binding: HIGHER_CPU, MID_CPU, and NO_BIND.
>
> The rule for determining the core binding mode is based on the frequency of CPU cores instead of the CPU architecture.
>
> HIGHER_CPU: indicates that threads in the thread pool are preferentially bound to the core with the highest frequency. The first thread is bound to the core with the highest frequency, the second thread is bound to the core with the second highest frequency, and so on.
>
> Mediumcores are defined based on experience. By default, mediumcores are with the third and fourth highest frequency. Mediumcore first indicates that threads are bound to mediumcores preferentially. When there are no available mediumcores, threads are bound to small cores.

### Resizing the Input Dimension

When using MindSpore Lite for inference, if you need to resize the input shape, you can call the [resize](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#resize) API of [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html) to reset the shape of the input tensor after creating a session and building a graph.

> Some networks do not support variable dimensions. As a result, an error message is displayed and the model exits unexpectedly. For example, the model contains the MatMul operator, one input tensor of the MatMul operator is the weight, and the other input tensor is the input. If a variable dimension API is called, the input tensor does not match the shape of the weight tensor. As a result, the inference fails.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L164) demonstrates how to perform [resize](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#resize) on the input tensor of MindSpore Lite:

```java
List<MSTensor> inputs = session.getInputs();
int[][] dims = {{1, 300, 300, 3}};
bool ret = session.resize(inputs, dims);
```

### Parallel Sessions

MindSpore Lite supports parallel inference of multiple [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html). The thread pool and memory pool of each [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) are independent. However, multiple threads cannot call the [runGraph](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#rungraph) API of a single [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html#litesession) at the same time.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L220) demonstrates how to infer multiple [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html) in parallel:

```java
session1 = createLiteSession(false);
if (session1 != null) {
    session1Compile = true;
} else {
    Toast.makeText(getApplicationContext(), "session1 Compile Failed.",
            Toast.LENGTH_SHORT).show();
}
session2 = createLiteSession(true);
if (session2 != null) {
    session2Compile = true;
} else {
    Toast.makeText(getApplicationContext(), "session2 Compile Failed.",
            Toast.LENGTH_SHORT).show();
}
...
if (session1Finish && session1Compile) {
    new Thread(new Runnable() {
        @Override
        public void run() {
            session1Finish = false;
            runInference(session1);
            session1Finish = true;
        }
    }).start();
}

if (session2Finish && session2Compile) {
    new Thread(new Runnable() {
        @Override
        public void run() {
            session2Finish = false;
            runInference(session2);
            session2Finish = true;
        }
    }).start();
}
```

MindSpore Lite does not support multi-thread parallel execution of inference for a single [LiteSession](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html). Otherwise, the following error information is displayed:

```text
ERROR [mindspore/lite/src/lite_session.cc:297] RunGraph] 10 Not support multi-threading
```

### Viewing Logs

If an exception occurs during inference, you can view logs to locate the fault. For the Android platform, use the `Logcat` command line to view the MindSpore Lite inference log information and use `MS_LITE` to filter the log information.

```bash
logcat -s "MS_LITE"
```

### Obtaining the Version Number

MindSpore Lite provides the [Version](https://www.mindspore.cn/lite/api/en/master/api_java/lite_session.html) method to obtain the version number, which is included in the `com.mindspore.lite.Version` header file. You can call this method to obtain the version number of MindSpore Lite.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L215) demonstrates how to obtain the version number of MindSpore Lite:

```java
import com.mindspore.lite.Version;
String version = Version.version();
```
