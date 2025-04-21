# Model Inference (Java)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/infer/runtime_java.md)

## Overview

After the model is converted into a `.ms` model by using the MindSpore Lite model conversion tool, the inference process can be performed in Runtime. For details, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html). This tutorial describes how to use the [Java API](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/index.html) to perform inference.

If MindSpore Lite is used in an Android project, you can use [C++ API](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/index.html) or [Java API](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/index.html) to run the inference framework. Compared with C++ APIs, Java APIs can be directly called in the Java class. Users do not need to implement the code at the JNI layer, which is more convenient. To run the MindSpore Lite inference framework, perform the following steps:

1. Load the model(optional): Read the `.ms` model converted by the model conversion tool introduced in [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html) from the file system.
2. Create a configuration context: Create a configuration context [MSContext](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#mscontext) to save some basic configuration parameters required by a model to guide graph build and execution, including `deviceType` (device type), `threadNum` (number of threads), `cpuBindMode` (CPU core binding mode), and `enable_float16` (whether to preferentially use the float16 operator).
3. Build a graph: Before building a graph, the [build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#build) API of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) needs to be called to build the graph, including graph partition and operator selection and scheduling. This takes a long time. Therefore, it is recommended that with [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) created each time, one graph be built. In this case, the inference will be performed for multiple times.
4. Input data: Before the graph is performed, data needs to be filled in to the `Input Tensor`.
5. Perform inference: Use the [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) of the [predict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#predict) to perform model inference.
6. Obtain the output: After the graph execution is complete, you can obtain the inference result by `outputting the tensor`.
7. Release the memory: If the MindSpore Lite inference framework is not required, release the created [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model).

![img](../images/lite_runtime.png)

> For details about the calling process of MindSpore Lite inference, see [Experience Java Simple Inference Demo](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/quick_start_java.html).

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

Before performing model inference, MindSpore Lite needs to load the `.ms` model converted by the model conversion tool from the file system and parse the model.

The following sample code reads the model file from specified file path.

```java
// Load the .ms model.
MappedByteBuffer byteBuffer = null;
try {
    fc = new RandomAccessFile(fileName, "r").getChannel();
    byteBuffer = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size()).load();
} catch (IOException e) {
    e.printStackTrace();
}
```

## Creating a Configuration Context

Create the configuration context [MSContext](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#mscontext) to save some basic configuration parameters required by the session to guide graph build and execution. Configure the number of threads, thread affinity and whether to enable heterogeneous parallel inference via the [init](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#init) interface. MindSpore Lite has a built-in thread pool shared by processes. During inference, `threadNum` is used to specify the maximum number of threads in the thread pool. The default value is 2.

MindSpore Lite supports heterogeneous inference. The preferred backend for inference is specified by `deviceType` of [AddDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo). Currently, CPU, GPU and NPU are supported. During graph build, operator selection and scheduling are performed based on the preferred backend.If the backend supports Float16, you can use the Float16 operator first by setting `isEnableFloat16` to `true`. If it is an NPU backend, you can also set the NPU frequency value. The default frequency value is 3, and can be set to 1 (low power consumption), 2 (balanced), 3 (high performance), and 4 (extreme performance).

### Configuring the CPU Backend

If the backend to be performed is a CPU, you need to configure [addDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo) after `MSContext` is inited. In addition, the CPU supports the setting of the core binding mode and whether to preferentially use the float16 operator.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L59) demonstrates how to create a CPU backend, set the CPU core binding mode to large-core priority, and enable float16 inference:

```java
MSContext context = new MSContext();
context.init(2, CpuBindMode.HIGHER_CPU);
context.addDeviceInfo(DeviceType.DT_CPU, true);
```

> Float16 takes effect only when the CPU is of the ARM v8.2 architecture. Other models and x86 platforms that are not supported are automatically rolled back to float32.

### Configuring the GPU Backend

If the backend to be performed is heterogeneous inference based on CPU and GPU, you need to add successively [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_GPUDeviceInfo.html) and [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_CPUDeviceInfo.html) when call [addDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo), GPU inference will be used first after configuration. In addition, if enable_float16 is set to true, both the GPU and CPU preferentially use the float16 operator.

The following sample code demonstrates how to create the CPU and GPU heterogeneous inference backend and how to enable float16 inference for the GPU.

```java
MSContext context = new MSContext();
context.init(2, CpuBindMode.MID_CPU);
context.addDeviceInfo(DeviceType.DT_GPU, true);
context.addDeviceInfo(DeviceType.DT_CPU, true);
```

> Currently, the GPU can run only on Android mobile devices. Therefore, only the `AAR` library can be run.

### Configuring the NPU Backend

If the backend to be performed is heterogeneous inference based on CPU and GPU, you need to add successively [KirinNPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_KirinNPUDeviceInfo.html) and [CPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_CPUDeviceInfo.html) when call [addDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo), NPU inference will be used first after configuration. In addition, if enable_float16 is set to true, both the NPU and CPU preferentially use the float16 operator.

The following sample code demonstrates how to create the CPU and NPU heterogeneous inference backend and how to enable float16 inference for the NPU.KirinNPUDeviceInfo frequency can be set by `NPUFrequency`.

```java
MSContext context = new MSContext();
context.init(2, CpuBindMode.MID_CPU);
context.addDeviceInfo(DeviceType.DT_NPU, true, 3);
context.addDeviceInfo(DeviceType.DT_CPU, true);
```

## Loading and Compiling a Model

When using MindSpore Lite to perform inference, [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) is the main entrance of inference, and the model can be realized through Model Loading, model compilation and model execution. Using the [MSContext](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#init) created in the previous step, call the composite [build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#build) interface to implement model loading and model compilation.

The following sample code demonstrates how to load and compile a model.

```java
Model model = new Model();
boolean ret = model.build(filePath, ModelType.MT_MINDIR, msContext);
```

## Inputting Data

MindSpore Lite Java APIs provide the `getInputsByTensorName` and `getInputs` methods to obtain the input tensor. Both the `byte[]` and `ByteBuffer` data types are supported. You can set the data of the input tensor by calling [setData](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#setdata).

1. Use the [getInputsByTensorName](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#getinputsbytensorname) method to obtain the tensor connected to the input node from the model input tensor based on the name of the model input tensor. The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L151) demonstrates how to call the `getInputByTensorName` function to obtain the input tensor and fill in data.

    ```java
    MSTensor inputTensor = model.getInputByTensorName("2031_2030_1_construct_wrapper:x");
    // Set Input Data.
    inputTensor.setData(inputData);
    ```

2. Use the [getInputs](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#getinputs) method to directly obtain the vectors of all model input tensors. The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L113) demonstrates how to call `getInputs` to obtain the input tensors and fill in the data.

    ```java
    List<MSTensor> inputs = model.getInputs();
    MSTensor inputTensor = inputs.get(0);
    // Set Input Data.
    inputTensor.setData(inputData);
    ```

## Executing Inference

After MindSpore Lite builds a model, it can call the [predict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#predict) function of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) to perform model inference.

The following sample code demonstrates how to call `predict` to perform inference.

```java
// Run graph to infer results.
boolean ret = model.predict();
```

## Obtaining the Output

After performing inference, MindSpore Lite can output a tensor to obtain the inference result. MindSpore Lite provides three methods to obtain the output [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html) of a model and supports the [getByteData](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#getbytedata), [getFloatData](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#getfloatdata), [getIntData](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#getintdata) and [getLongData](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#getlongdata) methods to obtain the output data.

1. Use the [getOutputs](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#getoutputs) method to directly obtain the list of all model output [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#mstensor). The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L191) demonstrates how to call `getOutputs` to obtain the output tensor.

    ```java
    List<MSTensor> outTensors = model.getOutputs();
    ```

2. Use the [getOutputsByNodeName](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#getoutputsbynodename) method to obtain the vector of the tensor connected to the model output [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#mstensor) based on the name of the model output node. The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L175) demonstrates how to call `getOutputByTensorName` to obtain the output tensor.

    ```java
    MSTensor outTensor = model.getOutputsByNodeName("Default/head-MobileNetV2Head/Softmax-op204");
    // Apply infer results.
    ...
    ```

3. Use the [getOutputByTensorName](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#getoutputbytensorname) method to obtain the model output [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#mstensor) based on the name of the model output tensor. The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L182) demonstrates how to call `getOutputByTensorName` to obtain the output tensor.

    ```java
    MSTensor outTensor = model.getOutputByTensorName("Default/head-MobileNetV2Head/Softmax-op204");
    // Apply infer results.
    ...
    ```

## Releasing the Memory

If the MindSpore Lite inference framework is not required, you need to release the created Model. The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L204) demonstrates how to release the memory before the program ends.

```java
model.free();
```

## Advanced Usage

### Resizing the Input Dimension

When using MindSpore Lite for inference, if you need to resize the input shape, you can call the [resize](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#resize) API of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html) to reset the shape of the input tensor after building a model.

> Some networks do not support variable dimensions. As a result, an error message is displayed and the model exits unexpectedly. For example, the model contains the MatMul operator, one input tensor of the MatMul operator is the weight, and the other input tensor is the input. If a variable dimension API is called, the input tensor does not match the shape of the weight tensor. As a result, the inference fails.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L164) demonstrates how to perform [resize](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#resize) on the input tensor of MindSpore Lite:

```java
List<MSTensor> inputs = session.getInputs();
int[][] dims = {{1, 300, 300, 3}};
bool ret = model.resize(inputs, dims);
```

### Viewing Logs

If an exception occurs during inference, you can view logs to locate the fault. For the Android platform, use the `Logcat` command line to view the MindSpore Lite inference log information and use `MS_LITE` to filter the log information.

```bash
logcat -s "MS_LITE"
```

### Obtaining the Version Number

MindSpore Lite provides the [Version](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html) method to obtain the version number, which is included in the `com.mindspore.lite.Version` header file. You can call this method to obtain the version number of MindSpore Lite.

The following sample code from [MainActivity.java](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L215) demonstrates how to obtain the version number of MindSpore Lite:

```java
import com.mindspore.lite.config.Version;
String version = Version.version();
```
