# Experiencing Java Simplified Inference Demo

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/infer/quick_start_java.md)

## Overview

This tutorial provides an example program for MindSpore Lite to perform inference. It demonstrates the basic process of performing inference on the device side using [MindSpore Lite Java API](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/index.html) by random inputting data, executing inference, and printing the inference result. You can quickly understand how to use the Java APIs related to inference on MindSpore Lite. In this tutorial, the randomly generated data is used as the input data to perform the inference on the MobileNetV2 model and print the output data. The code is stored in the [mindspore/lite/examples/quick_start_java](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/quick_start_java) directory.

The MindSpore Lite inference steps are as follows:

1. Load the model(optional): Read the `.ms` model converted by the [model conversion tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html) from the file system.
2. Create and configure context: Create a configuration context [MSContext](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html#mscontext) to save some basic configuration parameters required by a session to guide graph build and execution. including `deviceType` (device type), `threadNum` (number of threads), `cpuBindMode` (CPU binding mode), and `enable_float16` (whether to preferentially use the float16 operator).
3. Build a graph: Before building a graph, the [build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#build) interface of [model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) needs to be called to build the graph, including subgraph partition and operator selection and scheduling. This takes a long time. Therefore, it is recommended that with one [model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) created, one graph be built. In this case, the inference will be performed for multiple times.
4. Input data: Before the graph is executed, data needs to be filled in the `Input Tensor`.
5. Perform inference: Use the [predict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#predict) of the [model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model) to perform model inference.
6. Obtain the output: After the graph execution is complete, you can obtain the inference result by `outputting the tensor`.
7. Release the memory: If the MindSpore Lite inference framework is not required, release the created [model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html#model).

![img](../images/lite_runtime.png)

> To view the advanced usage of MindSpore Lite, see [Using Runtime to Perform Inference (Java)](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/infer/runtime_java.html).

## Building and Running

- Environment requirements
    - System environment: Linux x86_64 (Ubuntu 18.04.02LTS is recommended.)
    - Build dependency:
        - [Git](https://git-scm.com/downloads) >= 2.28.0
        - [Maven](https://maven.apache.org/download.cgi) >= 3.3
        - [OpenJDK](https://openjdk.java.net/install/) 1.8 to 1.15

- Build

  Run the [build script](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/quick_start_java/build.sh) in the `mindspore/lite/examples/quick_start_java` directory to automatically download the MindSpore Lite inference framework library and model files and build the Demo.

  ```bash
  bash build.sh
  ```

  > If the MindSpore Lite inference framework fails to be downloaded, manually download the MindSpore Lite model inference framework [mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) whose hardware platform is CPU and operating system is Ubuntu-x64. Decompress the package and copy `runtime/lib/mindspore-lite-java.jar` file to the `mindspore/lite/examples/quick_start_java/lib` directory.
  >
  > If the MobileNetV2 model fails to be downloaded, manually download the model file [mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms) and copy it to the `mindspore/lite/examples/quick_start_java/model/` directory.
  >
  > After manually downloading and placing the file in the specified location, you need to execute the build.sh script again to complete the compilation.

- Inference

  After the build, go to the `mindspore/lite/examples/quick_start_java/target` directory and run the following command to experience MindSpore Lite inference on the MobileNetV2 model:

  ```bash
  java -classpath .:./quick_start_java.jar:../lib/mindspore-lite-java.jar  com.mindspore.lite.demo.Main ../model/mobilenetv2.ms
  ```

  After the execution, the following information is displayed, including the tensor name, tensor size, number of output tensors, and the first 50 pieces of data.

  ```text
  out tensor shape: [1,1000,] and out data: 5.4091015E-5 4.030303E-4 3.032344E-4 4.0029243E-4 2.2730739E-4 8.366581E-5 2.629827E-4 3.512394E-4 2.879536E-4 1.9557697E-4xxxxxxxxxx MindSpore Lite 1.1.0out tensor shape: [1,1000,] and out data: 5.4091015E-5 4.030303E-4 3.032344E-4 4.0029243E-4 2.2730739E-4 8.366581E-5 2.629827E-4 3.512394E-4 2.879536E-4 1.9557697E-4tensor name is:Default/Sigmoid-op204 tensor size is:2000 tensor elements num is:500output data is:3.31223e-05 1.99382e-05 3.01624e-05 0.000108345 1.19685e-05 4.25282e-06 0.00049955 0.000340809 0.00199094 0.000997094 0.00013585 1.57605e-05 4.34131e-05 1.56114e-05 0.000550819 2.9839e-05 4.70447e-06 6.91601e-06 0.000134483 2.06795e-06 4.11612e-05 2.4667e-05 7.26248e-06 2.37974e-05 0.000134513 0.00142482 0.00011707 0.000161848 0.000395011 3.01961e-05 3.95325e-05 3.12398e-06 3.57709e-05 1.36277e-06 1.01068e-05 0.000350805 5.09019e-05 0.000805241 6.60321e-05 2.13734e-05 9.88654e-05 2.1991e-06 3.24065e-05 3.9479e-05 4.45178e-05 0.00205024 0.000780899 2.0633e-05 1.89997e-05 0.00197261 0.000259391
  ```

## Model Loading(optional)

Read the MindSpore Lite model from the file system.

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

## Model Build

Model build includes context configuration creation and model compilation. current graph build support file and mappedbytebuffer format. The following sample code describes model compilation by reading from a file.

```java
private static boolean compile(String modelPath) {
    MSContext context = new MSContext();
    // use default param init context
    context.init();
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

## Model Inference

Model inference includes data input, inference execution, and output obtaining. In this example, the input data is randomly generated, and the output result is printed after inference.

```java
private static boolean run() {
    MSTensor inputTensor = model.getInputByTensorName("graph_input-173");
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
    MSTensor outTensor = model.getOutputByTensorName("Softmax-65");

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

## Memory Release

If the MindSpore Lite inference framework is not required, release the created `model`.

```java
// Delete model buffer.
model.free();
```
