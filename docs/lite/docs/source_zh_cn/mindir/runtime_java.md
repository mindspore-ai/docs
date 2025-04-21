# 使用Java接口执行云侧推理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/mindir/runtime_java.md)

## 概述

通过[MindSpore Lite模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool.html)转换成`.mindir`模型后，即可在Runtime中执行模型的推理流程。本教程介绍如何使用[JAVA接口](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/index.html)执行云侧推理。

相比C++ API，Java API可以直接在Java Class中调用，用户无需实现JNI层的相关代码，具有更好的便捷性。运行MindSpore Lite推理框架主要包括以下步骤：

1. 模型读取：通过MindSpore导出MindIR模型，或者由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool.html)转换获得MindIR模型。
2. 创建配置上下文：创建配置上下文[MSContext](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html#mscontext)，保存需要的一些基本配置参数，用于指导模型编译和模型执行，包括设备类型、线程数、绑核模式和使能fp16混合精度推理。
3. 模型创建、加载与编译：执行推理之前，需要调用[Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#model)的[build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#build)接口进行模型加载和模型编译，目前支持加载文件和MappedByteBuffer两种方式。模型加载阶段将文件或者buffer解析成运行时的模型。
4. 输入数据：模型执行之前需要向输入Tensor中填充数据。
5. 执行推理：使用[Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#model)的[predict](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#predict)方法进行模型推理。
6. 获得输出：图执行结束之后，可以通过输出Tensor得到推理结果。
7. 释放内存：无需使用MindSpore Lite推理框架的时候，需要释放已创建的[Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#model)。

![img](../images/lite_runtime.png)

## 引用MindSpore Lite Java库

### Linux 项目引用JAR库

采用`Maven`作为构建工具时，可将`mindspore-lite-java.jar`拷贝到根目录下的`lib`目录，并在`pom.xml`中增加jar包的依赖。

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

## 模型路径

通过MindSpore Lite进行模型推理时，需要获取文件系统中由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool.html)转换得到的`.mindir`模型文件的路径。

## 创建配置上下文

创建配置上下文[MSContext](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html#mscontext)，保存会话所需的一些基本配置参数，用于指导图编译和图执行。通过[init](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html#init)接口配置线程数，线程亲和性，以及是否开启异构并行推理。MindSpore Lite内置一个进程共享的线程池，推理时通过`threadNum`指定线程池的最大线程数，默认为2线程。

MindSpore Lite推理时的后端可调用[AddDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo)接口中的`deviceType`指定，目前支持CPU、GPU和Ascend。在进行图编译时，会根据主选后端进行算子选型调度。如果后端支持float16，可通过设置`isEnableFloat16`为`true`后，优先使用float16算子。

### 配置使用CPU后端

当需要执行的后端为CPU时，`MSContext`初始化后需要在[addDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo)中`DeviceType.DT_CPU`，同时CPU支持设置绑核模式以及是否优先使用float16算子。

下面演示如何创建CPU后端，同时设定线程数为2、CPU绑核模式为大核优先并且使能float16推理，关闭并行：

```java
MSContext context = new MSContext();
context.init(2, CpuBindMode.HIGHER_CPU);
context.addDeviceInfo(DeviceType.DT_CPU, true);
```

### 配置使用GPU后端

当需要执行的后端为GPU时，`MSContext`创建后需要在[addDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo)中添加[GPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#gpudeviceinfo)。如果使能float16推理，GPU会优先使用float16算子。

下面代码演示了如何创建GPU推理后端：

```java
MSContext context = new MSContext();
context.init();
context.addDeviceInfo(DeviceType.DT_GPU, true);
```

### 配置使用Ascend后端

当需要执行的后端为Ascend时，`MSContext`初始化后需要在[addDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html#adddeviceinfo)中添加[AscendDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#ascenddeviceinfo)。

下面演示如何创建Ascend后端：

```java
MSContext context = new MSContext();
context.init();
context.addDeviceInfo(DeviceType.DT_ASCEND, false, 0);
```

## 模型创建加载与编译

使用MindSpore Lite执行推理时，[Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#model)是推理的主入口，通过Model可以实现模型加载、模型编译和模型执行。采用上一步创建得到的[MSContext](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html#init)，调用Model的复合[build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#build)接口来实现模型加载与模型编译。

下面演示了Model创建、加载与编译的过程：

```java
Model model = new Model();
boolean ret = model.build(filePath, ModelType.MT_MINDIR, msContext);
```

## 输入数据

MindSpore Lite Java接口提供`getInputsByTensorName`以及`getInputs`两种方法获得输入Tensor，同时支持`byte[]`或者`ByteBuffer`两种类型的数据，通过[setData](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mstensor.html#setdata)设置输入Tensor的数据。

1. 使用[getInputsByTensorName](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#getinputsbytensorname)方法，根据模型输入Tensor的名称来获取模型输入Tensor中连接到输入节点的Tensor，下面演示如何调用`getInputsByTensorName`获得输入Tensor并填充数据。

    ```java
    MSTensor inputTensor = model.getInputsByTensorName("2031_2030_1_construct_wrapper:x");
    // Set Input Data.
    inputTensor.setData(inputData);
    ```

2. 使用[getInputs](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#getinputs)方法，直接获取所有的模型输入Tensor的vector，下面演示如何调用`getInputs`获得输入Tensor并填充数据。

    ```java
    List<MSTensor> inputs = model.getInputs();
    MSTensor inputTensor = inputs.get(0);
    // Set Input Data.
    inputTensor.setData(inputData);
    ```

## 执行推理

MindSpore Lite在模型编译以后，即可调用[Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#model)的[predict](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#predict)方法执行模型推理。

下面示例代码演示调用`predict`执行推理。

```java
// Run graph to infer results.
boolean ret = model.predict();
```

## 获得输出

MindSpore Lite在执行完推理后，可以通过输出Tensor得到推理结果。MindSpore Lite提供三种方法来获取模型的输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mstensor.html)，同时支持[getByteData](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mstensor.html#getbytedata)、[getFloatData](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mstensor.html#getfloatdata)、[getIntData](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mstensor.html#getintdata)、[getLongData](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mstensor.html#getlongdata)四种方法获得输出数据。

1. 使用[getOutputs](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#getoutputs)方法，直接获取所有的模型输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mstensor.html#mstensor)的列表。下面演示如何调用`getOutputs`获得输出Tensor列表。

    ```java
    List<MSTensor> outTensors = model.getOutputs();
    ```

2. 使用[getOutputsByNodeName](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#getoutputsbynodename)方法，根据模型输出节点的名称来获取模型输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mstensor.html#mstensor)中连接到该节点的Tensor的vector。下面演示如何调用`getOutputByTensorName`获得输出Tensor。

    ```java
    MSTensor outTensor = model.getOutputsByNodeName("Default/head-MobileNetV2Head/Softmax-op204");
    // Apply infer results.
    ...
    ```

3. 使用[getOutputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#getoutputbytensorname)方法，根据模型输出Tensor的名称来获取对应的模型输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mstensor.html#mstensor)。下面演示如何调用`getOutputByTensorName`获得输出Tensor。

    ```java
    MSTensor outTensor = model.getOutputByTensorName("Default/head-MobileNetV2Head/Softmax-op204");
    // Apply infer results.
    ...
    ```

## 释放内存

无需使用MindSpore Lite推理框架时，需要释放已经创建的Model，下列演示如何在程序结束前进行内存释放。

```java
model.free();
```

## 高级用法

### 输入维度Resize

使用MindSpore Lite进行推理时，如果需要对输入的shape进行Resize，则可以在模型编译`build`之后调用[Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html)的[Resize](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#resize)接口，对输入的Tensor重新设置shape。

> 某些网络是不支持可变维度，会提示错误信息后异常退出，比如，模型中有MatMul算子，并且MatMul的一个输入Tensor是权重，另一个输入Tensor是输入时，调用可变维度接口会导致输入Tensor和权重Tensor的Shape不匹配，最终导致推理失败。

下面演示如何对MindSpore Lite的输入Tensor进行[Resize](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html#resize)：

```java
List<MSTensor> inputs = model.getInputs();
int[][] dims = {{1, 300, 300, 3}};
bool ret = model.resize(inputs, dims);
```

### 查看日志

当推理出现异常的时候，可以通过查看日志信息来定位问题。

### 获取版本号

MindSpore Lite提供了[Version](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html)方法可以获取版本号，包含在`com.mindspore.lite.config.Version`头文件中，调用该方法可以得到当前MindSpore Lite的版本号。

下面演示如何获取MindSpore Lite的版本号：

```java
import com.mindspore.lite.config.Version;
String version = Version.version();
```
