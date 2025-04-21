# Using Java Interface to Perform Cloud-side Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/mindir/runtime_java.md)

## Overview

After converting the `.mindir` model by [MindSpore Lite model conversion tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool.html), you can execute the inference process of the model in Runtime. This tutorial describes how to perform cloud-side inference by using the [JAVA interface](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/index.html).

Compared with C++ API, Java API can be called directly in Java Class, and users do not need to implement the code related to JNI layer, with better convenience. Running MindSpore Lite inference framework mainly consists of the following steps:

1. Model reading: Export MindIR model via MindSpore or get MindIR model by [model conversion tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool.html).
2. Create configuration context: Create a configuration context [MSContext](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#mscontext) and save some basic configuration parameters used to guide model compilation and model execution, including device type, number of threads, CPU pinning, and enabling fp16 mixed precision inference.
3. Model creation, loading and compilation: Before executing inference, you need to call [build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#build) interface of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) for model loading and model compilation. Both loading files and MappedByteBuffer are currently supported. The model loading phase parses the file or buffer into a runtime model.
4. Input data: The model needs to be padded with data from the input Tensor before execution.
5. Execute inference: Use [predict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#predict) of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) method for model inference.
6. Obtain the output: After the graph execution, the inference result can be obtained by outputting the Tensor.
7. Release memory: When there is no need to use MindSpore Lite inference framework, you need to release the created [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model).

![img](../images/lite_runtime.png)

## Reference to MindSpore Lite Java Library

### Linux Project References to JAR Library

When using `Maven` as a build tool, you can copy `mindspore-lite-java.jar` to the `lib` directory in the root directory and add the jar package dependencies in `pom.xml`.

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

## Model Path

To perform model inference with MindSpore Lite, you need to get the path of the `.mindir` model file in the file system converted by [Model Conversion Tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool.html).

## Creating Configuration Context

Create a configuration context [MSContext](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#mscontext) and save some basic configuration parameters required for the session, which is used to guide graph compilation and graph execution. Configure the number of threads, thread affinity and whether to enable heterogeneous parallel inference via the [init](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#init) interface. MindSpore Lite has a built-in thread pool shared by processes. The maximum number of threads in the pool is specified by `threadNum` when inference, and the default is 2 threads.

The backend of MindSpore Lite inference can call `deviceType` in the [AddDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo) interface to specify, currently supporting CPU, GPU and Ascend. When graph compilation is performed, the operator selection is scheduled based on the main selection backend. If the backend supports float16, float16 operator can be used in preference by setting `isEnableFloat16` to `true`.

### Configuring to Use the CPU Backend

When the backend to be executed is CPU, `MSContext` needs to be initialized in `DeviceType.DT_CPU` of [addDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo), while the CPU supports setting the CPU pinning mode and whether to use float16 operator in preference.

The following demonstrates how to create a CPU backend, set the number of threads to 2, set the CPU pinning mode to large core priority and enable float16 inference, and turn off parallelism:

```java
MSContext context = new MSContext();
context.init(2, CpuBindMode.HIGHER_CPU);
context.addDeviceInfo(DeviceType.DT_CPU, true);
```

### Configuring to Use the GPU Backend

When the backend to be executed is GPU, after `MSContext` is created, you need to add [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_GPUDeviceInfo.html#class-gpudeviceinfo) in the [addDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo). If float16 inference is enabled, the GPU will use the float16 operator in preference.

The following code demonstrates how to create a GPU inference backend:

```java
MSContext context = new MSContext();
context.init();
context.addDeviceInfo(DeviceType.DT_GPU, true);
```

### Configuring to Use the Ascend Backend

When the backend to be executed is Ascend, after `MSContext` is created, you need to add [AscendDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#ascenddeviceinfo) in the [addDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo).

The following demonstrates how to create an Ascend backend:

```java
MSContext context = new MSContext();
context.init();
context.addDeviceInfo(DeviceType.DT_ASCEND, false, 0);
```

## Model Creation, Loading and Compilation

When using MindSpore Lite to perform inference, [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) is the main entry for inference. Model loading, compilation and execution are implemented through Model. Using the [MSContext](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#init) created in the previous step, call the compound [build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#build) interface of Model to implement model loading and model compilation.

The following demonstrates the process of Model creation, loading and compilation:

```java
Model model = new Model();
boolean ret = model.build(filePath, ModelType.MT_MINDIR, msContext);
```

## Inputting the Data

MindSpore Lite Java interface provides `getInputsByTensorName` and `getInputs` methods to get the input Tensor, and supports `byte[]` or `ByteBuffer` types of data, set the input Tensor data by [setData](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#setdata).

1. Use [getInputsByTensorName](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#getinputsbytensorname) method. Obtain the Tensor connected to the input node in the model input Tensor based on the name of the model input Tensor. The following demonstrates how to call `getInputsByTensorName` to get the input Tensor and pad the data.

    ```java
    MSTensor inputTensor = model.getInputsByTensorName("2031_2030_1_construct_wrapper:x");
    // Set Input Data.
    inputTensor.setData(inputData);
    ```

2. Use [getInputs](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#getinputs) method and obtain all the model input Tensor vectors directly. The following demonstrates how to call `getInputs` to get the input Tensor and pad the data.

    ```java
    List<MSTensor> inputs = model.getInputs();
    MSTensor inputTensor = inputs.get(0);
    // Set Input Data.
    inputTensor.setData(inputData);
    ```

## Executing the Inference

MindSpore Lite can call [predict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#predict) of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) to execute model inference after the model is compiled.

The following sample code demonstrates calling `predict` to perform inference.

```java
// Run graph to infer results.
boolean ret = model.predict();
```

## Obtaining the Output

MindSpore Lite can get the inference result by outputting Tensor after performing inference. MindSpore Lite provides three methods to get the output of the model [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html), and also supports [getByteData](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#getbytedata), [getFloatData](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#getfloatdata), [getIntData](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#getintdata), [getLongData](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#getlongdata) four methods to get the output data.

1. Use the [getOutputs](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#getoutputs) method, get all the model to output list of [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#mstensor). The following demonstrates how to call `getOutputs` to get the list of output Tensor.

    ```java
    List<MSTensor> outTensors = model.getOutputs();
    ```

2. Use the [getOutputsByNodeName](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#getoutputsbynodename) method and get the vector of the Tensor connected to that node in the model output [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#mstensor) according to the name of model output node. The following demonstrates how to call ` getOutputByTensorName` to get the output Tensor.

    ```java
    MSTensor outTensor = model.getOutputsByNodeName("Default/head-MobileNetV2Head/Softmax-op204");
    // Apply infer results.
    ...
    ```

3. Use the [getOutputByTensorName](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#getoutputbytensorname) method to get the corresponding model output [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html#mstensor) based on the name of the model output Tensor. The following demonstrates how to call `getOutputByTensorName` to get the output Tensor.

    ```java
    MSTensor outTensor = model.getOutputByTensorName("Default/head-MobileNetV2Head/Softmax-op204");
    // Apply infer results.
    ...
    ```

## Releasing the Memory

When there is no need to use the MindSpore Lite inference framework, it is necessary to free the created models. The following demonstrates how to do the memory release before the end of the program.

```java
model.free();
```

## Advanced Usage

### Input Dimension Resize

When using MindSpore Lite for inference, if you need to Resize the input shape, you can call the[Resize](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#resize) of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html) to reset the shape of the input Tensor after the model compiles `build`.

> Some networks do not support variable dimensions and will exit abnormally after prompting an error message. For example, when there is a MatMul operator in the model and one input Tensor of MatMul is the weight and the other input Tensor is the input, calling the variable dimension interface will cause the Shape of the input Tensor and the weight Tensor to mismatch, which eventually fails the inference.

The following demonstrates how to [Resize](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#resize) the input Tensor of MindSpore Lite:

```java
List<MSTensor> inputs = model.getInputs();
int[][] dims = {{1, 300, 300, 3}};
bool ret = model.resize(inputs, dims);
```

### Viewing the Logs

When an exception occurs in inference, the problem can be located by viewing log information.

### Obtaining the Version Number

MindSpore Lite provides the [Version](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html) method to get the version number, which is included in the `com.mindspore.lite.config.Version` header file. This method is called to get the current version number of MindSpore Lite.

The following demonstrates how to get the version number of MindSpore Lite:

```java
import com.mindspore.lite.config.Version;
String version = Version.version();
```
