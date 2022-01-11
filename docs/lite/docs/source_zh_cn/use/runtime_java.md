# 使用Java接口执行推理

`Android` `Java` `推理应用` `模型加载` `数据准备` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/lite/docs/source_zh_cn/use/runtime_java.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## 概述

通过[MindSpore Lite模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r1.6/use/converter_tool.html)转换成`.ms`模型后，即可在Runtime中执行模型的推理流程。本教程介绍如何使用[JAVA接口](https://www.mindspore.cn/lite/api/zh-CN/r1.6/index.html)执行推理。

Android项目中使用MindSpore Lite，可以选择采用[C++ API](https://www.mindspore.cn/lite/api/zh-CN/r1.6/index.html)或者[Java API](https://www.mindspore.cn/lite/api/zh-CN/r1.6/index.html)运行推理框架。Java API与C++ API相比较而言，Java API可以直接在Java Class中调用，用户无需实现JNI层的相关代码，具有更好的便捷性。运行MindSpore Lite推理框架主要包括以下步骤：

1. 模型加载：从文件系统中读取由[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r1.6/use/converter_tool.html)转换得到的`.ms`模型，通过Model的[loadModel](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/model.html#loadmodel)导入模型。
2. 创建配置上下文：创建配置上下文[MSConfig](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/msconfig.html#msconfig)，保存会话所需的一些基本配置参数，用于指导图编译和图执行。主要包括`deviceType`：设备类型、`threadNum`：线程数、`cpuBindMode`：CPU绑定模式、`enable_float16`：是否优先使用Float16算子。
3. 创建会话：创建[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)，并调用[init](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#init)方法将上一步得到的[MSConfig](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/msconfig.html#msconfig)配置到会话中。
4. 图编译：在图执行前，需要调用[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)的[compileGraph](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#compilegraph)接口进行图编译，主要进行子图切分、算子选型调度。这部分会耗费较多时间，所以建议[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)创建一次，编译一次，多次执行。
5. 输入数据：图执行之前需要向输入Tensor中填充数据。
6. 执行推理：使用[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)的[runGraph](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#rungraph)进行模型推理。
7. 获得输出：图执行结束之后，可以通过输出Tensor得到推理结果。
8. 释放内存：无需使用MindSpore Lite推理框架的时候，需要释放已创建的[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)和[model](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/model.html#model)。

![img](../images/lite_runtime.png)

快速了解MindSpore Lite执行推理的完整调用流程，请参考[体验Java极简推理Demo](https://www.mindspore.cn/lite/docs/zh-CN/r1.6/quick_start/quick_start_java.html)。

## 引用MindSpore Lite Java库

### Linux X86项目引用JAR库

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

> 运行时需要将`libmindspore-lite.so`以及`libminspore-lite-jni.so`的所在路径添加到`java.library.path`。

### Android项目引用AAR库

采用`Gradle`作为构建工具时，首先将`mindspore-lite-{version}.aar`文件移动到目标module的`libs`目录，然后在目标module的`build.gradle`的`repositories`中添加本地引用目录，最后在`dependencies`中添加AAR的依赖，具体如下所示。

> 注意mindspore-lite-{version}是AAR的文件名，需要将{version}替换成对应版本信息。

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

## 加载模型

MindSpore Lite进行模型推理时，需要先从文件系统中加载模型转换工具转换后的`.ms`模型，并进行模型解析。Java的[model](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/model.html#model)类提供了2个[loadModel](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/model.html#loadmodel)接口，使其可以从`Assets`或其他文件路径中加载模型。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L217)将从`Assets`读取`mobilenetv2.ms`模型文件进行模型加载。

```java
// Load the .ms model.
Model model = new Model();
String modelPath = "mobilenetv2.ms";
boolean ret = model.loadModel(this.getApplicationContext(), modelPath);
```

>只有`AAR`库才支持从`Assert`加载模型文件的接口。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/quick_start_java/src/main/java/com/mindspore/lite/demo/Main.java#L128)将从`modelPath`路径读取模型文件进行模型加载。

```java
Model model = new Model();
boolean ret = model.loadModel(modelPath);
```

## 创建配置上下文

创建配置上下文[MSConfig](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/msconfig.html#msconfig)，保存会话所需的一些基本配置参数，用于指导图编译和图执行。

MindSpore Lite支持异构推理，推理时的主选后端由[MSConfig](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/msconfig.html#msconfig)的`deviceType`指定，目前支持CPU和GPU。在进行图编译时，会根据主选后端进行算子选型调度。

MindSpore Lite内置一个进程共享的线程池，推理时通过`threadNum`指定线程池的最大线程数，默认为2线程。

MindSpore Lite支持Float16算子的模式进行推理。`enable_float16`设置为`true`后，将会优先使用Float16算子。

### 配置使用CPU后端

当需要执行的后端为CPU时，`MSConfig`创建后需要在[init](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/msconfig.html#init)中配置`DeviceType.DT_CPU`，同时CPU支持设置绑核模式以及是否优先使用Float16算子。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L59)演示如何创建CPU后端，同时设定CPU绑核模式为大核优先并且使能Float16推理：

```java
MSConfig msConfig = new MSConfig();
boolean ret = msConfig.init(DeviceType.DT_CPU, 2, CpuBindMode.HIGHER_CPU, true);
```

> Float16需要CPU为ARM v8.2架构的机型才能生效，其他不支持的机型和x86平台会自动回退到Float32执行。

### 配置使用GPU后端

当需要执行的后端为CPU和GPU的异构推理时，`MSConfig`创建后需要在[init](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/msconfig.html#init)中配置`DeviceType.DT_GPU`，配置后将会优先使用GPU推理。同时是否优先使用Float16算子设置为true后，GPU和CPU都会优先使用Float16算子。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L69)演示如何创建CPU与GPU异构推理后端，同时GPU也设定使能Float16推理：

```java
MSConfig msConfig = new MSConfig();
boolean ret = msConfig.init(DeviceType.DT_GPU, 2, CpuBindMode.MID_CPU, true);
```

> 目前GPU只能在Android手机端侧运行，所以只有`AAR`库才能支持运行。

## 创建会话

[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)是推理的主入口，通过[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)可以进行图编译、图执行。创建[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)，并调用[init](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#init)方法将上一步得到[MSConfig](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/msconfig.html#msconfig)配置到会话中。[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)初始化之后，[MSConfig](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/msconfig.html#msconfig)将可以进行释放操作。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L86)演示如何创建`LiteSession`的方式：

```java
LiteSession session = new LiteSession();
boolean ret = session.init(msConfig);
msConfig.free();
```

## 图编译

在图执行前，需要调用[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)的[compileGraph](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#compilegraph)接口进行图编译，主要进行子图切分、算子选型调度。这部分会耗费较多时间，所以建议[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)创建一次，编译一次，多次执行。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L87)演示调用`CompileGraph`进行图编译。

```java
boolean ret = session.compileGraph(model);
```

## 输入数据

MindSpore Lite Java接口提供`getInputsByTensorName`以及`getInputs`两种方法获得输入Tensor，同时支持`byte[]`或者`ByteBuffer`两种类型的数据，通过[setData](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/mstensor.html#setdata)设置输入Tensor的数据。

1. 使用[getInputsByTensorName](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#getinputsbytensorname)方法，根据模型输入Tensor的名称来获取模型输入Tensor中连接到输入节点的Tensor，下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L151)演示如何调用`getInputsByTensorName`获得输入Tensor并填充数据。

    ```java
    MSTensor inputTensor = session.getInputsByTensorName("2031_2030_1_construct_wrapper:x");
    // Set Input Data.
    inputTensor.setData(inputData);
    ```

2. 使用[getInputs](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#getinputs)方法，直接获取所有的模型输入Tensor的vector，下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L113)演示如何调用`getInputs`获得输入Tensor并填充数据。

    ```java
    List<MSTensor> inputs = session.getInputs();
    MSTensor inputTensor = inputs.get(0);
    // Set Input Data.
    inputTensor.setData(inputData);
    ```

> MindSpore Lite的模型输入Tensor中的数据排布必须是`NHWC`。如果需要了解更多数据前处理过程，可参考[基于Java接口的Android应用开发](https://www.mindspore.cn/lite/docs/zh-CN/r1.6/quick_start/image_segmentation.html#id9)的对输入数据进行处理部分。

## 执行推理

MindSpore Lite会话在进行图编译以后，即可调用[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)的[runGraph](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#rungraph)执行模型推理。

下面示例代码演示调用`runGraph`执行推理。

```java
// Run graph to infer results.
boolean ret = session.runGraph();
```

## 获得输出

MindSpore Lite在执行完推理后，可以通过输出Tensor得到推理结果。MindSpore Lite提供三种方法来获取模型的输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/mstensor.html)，同时支持[getByteData](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/mstensor.html#getbytedata)、[getFloatData](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/mstensor.html#getfloatdata)、[getIntData](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/mstensor.html#getintdata)、[getLongData](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/mstensor.html#getlongdata)四种方法获得输出数据。

1. 使用[getOutputMapByTensor](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#getoutputmapbytensor)方法，直接获取所有的模型输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/mstensor.html#mstensor)的名称和[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/mstensor.html#mstensor)指针的一个map。下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L191)演示如何调用`getOutputMapByTensor`获得输出Tensor。

    ```java
    Map<String, MSTensor> outTensors = session.getOutputMapByTensor();

    Iterator<Map.Entry<String, MSTensor>> entries = outTensors.entrySet().iterator();
    while (entries.hasNext()) {
        Map.Entry<String, MSTensor> entry = entries.next();
        // Apply infer results.
        ...
    }
    ```

2. 使用[getOutputByNodeName](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#getoutputsbynodename)方法，根据模型输出节点的名称来获取模型输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/mstensor.html#mstensor)中连接到该节点的Tensor的vector。下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L175)演示如何调用`getOutputByTensorName`获得输出Tensor。

    ```java
    MSTensor outTensor = session.getOutputsByNodeName("Default/head-MobileNetV2Head/Softmax-op204");
    // Apply infer results.
    ...
    ```

3. 使用[getOutputByTensorName](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#getoutputbytensorname)方法，根据模型输出Tensor的名称来获取对应的模型输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/mstensor.html#mstensor)。下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L182)演示如何调用`getOutputByTensorName`获得输出Tensor。

    ```java
    MSTensor outTensor = session.getOutputByTensorName("Default/head-MobileNetV2Head/Softmax-op204");
    // Apply infer results.
    ...
    ```

## 释放内存

无需使用MindSpore Lite推理框架时，需要释放已经创建的LiteSession和Model，下列[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L204)演示如何在程序结束前进行内存释放。

```java
session.free();
model.free();
```

## 高级用法

### 优化运行内存大小

如果对运行时内存有较大的限制，图编译结束之后，调用[Model](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/model.html#model)的[freeBuffer](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/model.html#freebuffer)函数，释放MindSpore Lite Model中的MetaGraph，用于减小运行时的内存。一旦调用某个[Model](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/model.html#model)的[freeBuffer](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/model.html#freebuffer)后，该[Model](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/model.html#model)就不能再次图编译。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L241)演示如何调用`Model`的`freeBuffer`接口来释放`MetaGraph`减少运行时内存大小。

```java
// Compile graph.
ret = session.compileGraph(model);
...
// Note: when use model.freeBuffer(), the model can not be compiled.
model.freeBuffer();
```

### 绑核操作

MindSpore Lite内置线程池支持绑核、解绑操作，通过调用[bindThread](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#bindthread)接口，可以将线程池中的工作线程绑定到指定CPU核，用于性能分析。绑核操作与创建[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html)时用户指定的上下文有关，绑核操作会根据上下文中的绑核策略进行线程与CPU的亲和性设置。

需要注意的是，绑核是一个亲和性操作，不保证一定能绑定到指定的CPU核，会受到系统调度的影响。而且绑核后，需要在执行完代码后进行解绑操作。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L164)演示如何在执行推理时绑定大核优先。

```java
boolean ret = msConfig.init(DeviceType.DT_CPU, 2, CpuBindMode.HIGHER_CPU, true);
...
session.bindThread(true);
// Run Inference.
ret = session.runGraph();
session.bindThread(false);
```

> 绑核参数有三种选择：大核优先、中核优先以及不绑核。
>
> 判定大核和中核的规则其实是根据CPU核的频率而不是根据CPU的架构，对于没有大中小核之分的CPU架构，在该规则下也可以区分大核和中核。
>
> 绑定大核优先是指线程池中的线程从频率最高的核开始绑定，第一个线程绑定在频率最高的核上，第二个线程绑定在频率第二高的核上，以此类推。
>
> 对于中核优先，中核的定义是根据经验来定义的，默认设定中核是第三和第四高频率的核，当绑定策略为中核优先时，会优先绑定到中核上，当中核不够用时，会往小核上进行绑定。

### 输入维度Resize

使用MindSpore Lite进行推理时，如果需要对输入的shape进行Resize，则可以在已完成创建会话`CreateSession`与图编译`CompileGraph`之后调用[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html)的[resize](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#resize)接口，对输入的Tensor重新设置shape。

> 某些网络是不支持可变维度，会提示错误信息后异常退出，比如，模型中有MatMul算子，并且MatMul的一个输入Tensor是权重，另一个输入Tensor是输入时，调用可变维度接口会导致输入Tensor和权重Tensor的Shape不匹配，最终导致推理失败。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L164)演示如何对MindSpore Lite的输入Tensor进行[resize](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#resize)：

```java
List<MSTensor> inputs = session.getInputs();
int[][] dims = {{1, 300, 300, 3}};
bool ret = session.resize(inputs, dims);
```

### Session并行

MindSpore Lite支持多个[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html)并行推理，每个[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)的线程池和内存池都是独立的。但不支持多个线程同时调用单个[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#litesession)的[runGraph](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html#rungraph)接口。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L220)演示如何并行执行推理多个[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html)的过程：

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

MindSpore Lite不支持多线程并行执行单个[LiteSession](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html)的推理，否则会得到以下错误信息：

```text
ERROR [mindspore/lite/src/lite_session.cc:297] RunGraph] 10 Not support multi-threading
```

### 查看日志

当推理出现异常的时候，可以通过查看日志信息来定位问题。针对Android平台，采用`Logcat`命令行工具查看MindSpore Lite推理的日志信息，并利用`MS_LITE` 进行筛选。

```bash
logcat -s "MS_LITE"
```

### 获取版本号

MindSpore Lite提供了[Version](https://www.mindspore.cn/lite/api/zh-CN/r1.6/api_java/lite_session.html)方法可以获取版本号，包含在`com.mindspore.lite.Version`头文件中，调用该方法可以得到当前MindSpore Lite的版本号。

下面[示例代码](https://gitee.com/mindspore/mindspore/blob/r1.6/mindspore/lite/examples/runtime_java/app/src/main/java/com/mindspore/lite/demo/MainActivity.java#L215)演示如何获取MindSpore Lite的版本号：

```java
import com.mindspore.lite.Version;
String version = Version.version();
```
