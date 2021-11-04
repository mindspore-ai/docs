# 推理模型转换

`Windows` `Linux` `模型转换` `中级` `高级`

<!-- TOC -->

- [推理模型转换](#推理模型转换)
    - [概述](#概述)
    - [Linux环境使用说明](#linux环境使用说明)
        - [环境准备](#环境准备)
        - [目录结构](#目录结构)
        - [参数说明](#参数说明)
        - [使用示例](#使用示例)
    - [Windows环境使用说明](#windows环境使用说明)
        - [环境准备](#环境准备-1)
        - [目录结构](#目录结构-1)
        - [参数说明](#参数说明-1)
        - [使用示例](#使用示例-1)
    - [高级用法](#高级用法)
        - [Pass扩展](#pass扩展)
        - [算子InferShape扩展](#算子infershape扩展)
        - [示例演示](#示例演示)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/use/converter_tool.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore Lite提供离线转换模型功能的工具，支持多种类型的模型转换，转换后的模型可用于推理。命令行参数包含多种个性化选项，为用户提供方便的转换途径。

目前支持的输入格式有：MindSpore、TensorFlow Lite、Caffe、TensorFlow和ONNX。

通过转换工具转换成的`ms`模型，支持转换工具配套及更高版本的Runtime推理框架执行推理。

## Linux环境使用说明

### 环境准备

使用MindSpore Lite模型转换工具，需要进行如下环境准备工作。

- [编译](https://www.mindspore.cn/lite/docs/zh-CN/master/use/build.html)或[下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)模型转换工具。
- 将转换工具需要的动态链接库加入环境变量LD_LIBRARY_PATH。

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ```

    ${PACKAGE_ROOT_PATH}是编译或下载得到的包解压后的路径。

### 目录结构

```text
mindspore-lite-{version}-linux-x64
└── tools
    └── converter
        ├── include
        │   └── registry             # 自定义算子、模型解析、节点解析、转换优化注册头文件
        ├── converter                # 模型转换工具
        │   └── converter_lite       # 可执行程序
        └── lib                      # 转换工具依赖的动态库
            ├── libglog.so.0         # Glog的动态库
            ├── libmslite_converter_plugin.so  # 注册插件的动态库
            ├── libopencv_core.so.4.5          # OpenCV的动态库
            ├── libopencv_imgcodecs.so.4.5     # OpenCV的动态库
            └── libopencv_imgproc.so.4.5       # OpenCV的动态库
```

### 参数说明

MindSpore Lite模型转换工具提供了多种参数设置，用户可根据需要来选择使用。此外，用户可输入`./converter_lite --help`获取实时帮助。

下面提供详细的参数说明。

| 参数  |  是否必选   |  参数说明  | 取值范围 | 默认值 |
| -------- | ------- | ----- | --- | ---- |
| `--help` | 否 | 打印全部帮助信息。 | - | - |
| `--fmk=<FMK>`  | 是 | 输入模型的原始格式。 | MINDIR、CAFFE、TFLITE、TF、ONNX | - |
| `--modelFile=<MODELFILE>` | 是 | 输入模型的路径。 | - | - |
| `--outputFile=<OUTPUTFILE>` | 是 | 输出模型的路径，不需加后缀，可自动生成`.ms`后缀。 | - | - |
| `--weightFile=<WEIGHTFILE>` | 转换Caffe模型时必选 | 输入模型weight文件的路径。 | - | - |
| `--configFile=<CONFIGFILE>` | 否 | 1）可作为训练后量化配置文件路径；2）可作为扩展功能配置文件路径。  | - | - |
| `--fp16=<FP16>` | 否 | 设定在模型序列化时是否需要将Float32数据格式的权重存储为Float16数据格式。 | on、off | off |
| `--inputShape=<INPUTSHAPE>` | 否 | 设定模型输入的维度，输入维度的顺序和原始模型保持一致。对某些特定的模型可以进一步优化模型结构，但是转化后的模型将可能失去动态shape的特性。多个输入用`;`分割，同时加上双引号`""`。 | e.g.  "inTensorName_1: 1,32,32,4;inTensorName_2:1,64,64,4;" | - |
| `--inputDataFormat=<INPUTDATAFORMAT>` | 否 | 设定导出模型的输入format，只对4维输入有效。 | NHWC、NCHW | NHWC |
| `--decryptKey=<DECRYPTKEY>` | 否 | 设定用于加载密文MindIR时的密钥，密钥用十六进制表示，只对`fmk`为MINDIR时有效。 | - | - |
| `--decryptMode=<DECRYPTMODE>` | 否 | 设定加载密文MindIR的模式，只在指定了decryptKey时有效。 | AES-GCM、AES-CBC | AES-GCM |
| `--inputDataType=<INPUTDATATYPE>` | 否 | 设定量化模型输入tensor的data type。仅当模型输入tensor的量化参数（scale和zero point）齐备时有效。默认与原始模型输入tensor的data type保持一致。 | FLOAT32、INT8、UINT8、DEFAULT | DEFAULT |
| `--outputDataType=<OUTPUTDATATYPE>` | 否 | 设定量化模型输出tensor的data type。仅当模型输出tensor的量化参数（scale和zero point）齐备时有效。默认与原始模型输出tensor的data type保持一致。 | FLOAT32、INT8、UINT8、DEFAULT | DEFAULT |

> - 参数名和参数值之间用等号连接，中间不能有空格。
> - Caffe模型一般分为两个文件：`*.prototxt`模型结构，对应`--modelFile`参数；`*.caffemodel`模型权值，对应`--weightFile`参数。
> - `--fp16`的优先级很低，比如如果开启了量化，那么对于已经量化的权重，`--fp16`不会再次生效。总而言之，该选项只会在序列化时对模型中的Float32的权重生效。
> - `inputDataFormat`：一般在集成NCHW规格的三方硬件场景下(例如[集成NNIE使用说明](https://www.mindspore.cn/lite/docs/zh-CN/master/use/nnie.html#nnie))，设为NCHW比NHWC会有较明显的性能提升。在其他场景下，用户也可按需设置。
> - `configFile`配置文件采用`key=value`的方式定义相关参数，量化相关的配置参数详见[训练后量化](https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html)，扩展功能相关的配置参数详见[扩展配置](https://www.mindspore.cn/lite/docs/zh-CN/master/use/nnie.html#id6)。

### 使用示例

下面选取了几个常用示例，说明转换命令的使用方法。

- 以Caffe模型LeNet为例，执行转换命令。

   ```bash
   ./converter_lite --fmk=CAFFE --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
   ```

   本例中，因为采用了Caffe模型，所以需要模型结构、模型权值两个输入文件。再加上其他必需的fmk类型和输出路径两个参数，即可成功执行。

   结果显示为：

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

   这表示已经成功将Caffe模型转化为MindSpore Lite模型，获得新文件`lenet.ms`。

- 以MindSpore、TensorFlow Lite、TensorFlow和ONNX模型为例，执行转换命令。

    - MindSpore模型`model.mindir`

      ```bash
      ./converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model
      ```

     > 通过MindSpore v1.1.1之前版本导出的`MindIR`模型，建议采用对应版本的转换工具转换成`ms`模型。MindSpore v1.1.1及其之后的版本，转换工具会做前向兼容。

    - TensorFlow Lite模型`model.tflite`

      ```bash
      ./converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model
      ```

    - TensorFlow模型`model.pb`

      ```bash
      ./converter_lite --fmk=TF --modelFile=model.pb --outputFile=model
      ```

    - ONNX模型`model.onnx`

      ```bash
      ./converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model
      ```

   以上几种情况下，均显示如下转换成功提示，且同时获得`model.ms`目标文件。

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

> 训练后量化示例请参考<https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html>。

## Windows环境使用说明

### 环境准备

使用MindSpore Lite模型转换工具，需要进行如下环境准备工作。

- [编译](https://www.mindspore.cn/lite/docs/zh-CN/master/use/build.html)或[下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)模型转换工具。
- 将转换工具需要的动态链接库加入环境变量PATH。

  ```bash
  set PATH=%PACKAGE_ROOT_PATH%\tools\converter\lib;%PATH%
  ```

  ${PACKAGE_ROOT_PATH}是编译或下载得到的包解压后的路径。

### 目录结构

```text
mindspore-lite-{version}-win-x64
└── tools
    └── converter # 模型转换工具
        ├── converter
        │   └── converter_lite.exe    # 可执行程序
        └── lib
            ├── libgcc_s_seh-1.dll    # MinGW动态库
            ├── libglog.dll           # Glog的动态库
            ├── libmslite_converter_plugin.dll   # 注册插件的动态库
            ├── libmslite_converter_plugin.dll.a # 注册插件的动态库的链接文件
            ├── libssp-0.dll          # MinGW动态库
            ├── libstdc++-6.dll       # MinGW动态库
            └── libwinpthread-1.dll   # MinGW动态库
```

### 参数说明

参考Linux环境模型转换工具的[参数说明](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html#id3)。

### 使用示例

设置日志打印级别为INFO。

```bat
set GLOG_v=1
```

> 日志级别：0代表DEBUG，1代表INFO，2代表WARNING，3代表ERROR。

下面选取了几个常用示例，说明转换命令的使用方法。

- 以Caffe模型LeNet为例，执行转换命令。

   ```bat
   call converter_lite --fmk=CAFFE --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
   ```

   本例中，因为采用了Caffe模型，所以需要模型结构、模型权值两个输入文件。再加上其他必需的fmk类型和输出路径两个参数，即可成功执行。

   结果显示为：

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

   这表示已经成功将Caffe模型转化为MindSpore Lite模型，获得新文件`lenet.ms`。

- 以MindSpore、TensorFlow Lite、ONNX模型格式和感知量化模型为例，执行转换命令。

    - MindSpore模型`model.mindir`

      ```bat
      call converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model
      ```

      > 通过MindSpore v1.1.1之前版本导出的`MindIR`模型，建议采用对应版本的转换工具转换成`ms`模型。MindSpore v1.1.1及其之后的版本，转换工具会做前向兼容。

    - TensorFlow Lite模型`model.tflite`

      ```bat
      call converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model
      ```

    - TensorFlow模型`model.pb`

      ```bat
      call converter_lite --fmk=TF --modelFile=model.pb --outputFile=model
      ```

    - ONNX模型`model.onnx`

      ```bat
      call converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model
      ```

   以上几种情况下，均显示如下转换成功提示，且同时获得`model.ms`目标文件。

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

## 高级用法

转换工具仅在Linux环境下支持外部扩展功能，包括节点解析扩展、模型解析扩展以及图优化扩展。用户可以按需任意组合，以实现自己的意图。

> - 节点解析扩展：用户自定义模型中某一节点的解析过程，支持ONNX、CAFFE、TF、TFLITE。接口可参考[NodeParser](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_converter.html#nodeparser)。
> - 模型解析扩展：用户自定义模型的整个解析过程，支持ONNX、CAFFE、TF、TFLITE。接口可参考[ModelParser](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_converter.html#modelparser)。
> - 图优化扩展：模型解析之后，用户可自定义对图的优化过程。接口可参考[PassBase](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_registry.html#passbase)。
>
> 节点解析扩展需要依赖flatbuffers和protobuf及三方框架的序列化文件，并且flatbuffers和protobuf需要与发布件采用的版本一致，序列化文件需保证兼容发布件采用的序列化文件。发布件中不提供flatbuffers、protobuf及序列化文件，用户需自行编译，并生成序列化文件。用户可以从[MindSpore仓](https://gitee.com/mindspore/mindspore/tree/master)中获取[flabuffers](https://gitee.com/mindspore/mindspore/blob/master/cmake/external_libs/flatbuffers.cmake)、[probobuf](https://gitee.com/mindspore/mindspore/blob/master/cmake/external_libs/protobuf.cmake)、[ONNX原型文件](https://gitee.com/mindspore/mindspore/tree/master/third_party/proto/onnx)、[CAFFE原型文件](https://gitee.com/mindspore/mindspore/tree/master/third_party/proto/caffe)、[TF原型文件](https://gitee.com/mindspore/mindspore/tree/master/third_party/proto/tensorflow)和[TFLITE原型文件](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/tools/converter/parser/tflite/schema.fbs)。

本章节将通过MindSpore Lite转换工具扩展功能的示例程序，涵盖了Pass的创建全流程以及编译链接全流程，来使用户能够快速了解转换工具的图优化扩展功能的使用。

本章节以[add.tflite](https://download.mindspore.cn/model_zoo/official/lite/quick_start/add.tflite)模型为例。该模型仅包含一个简单的Add算子，通过扩展的Pass类，将Add算子转化为[Custom算子](https://www.mindspore.cn/lite/docs/zh-CN/master/use/register_kernel.html#custom)，最终输出Custom单算子模型。

相关代码放置在[mindspore/lite/examples/converter_extend](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/converter_extend)目录。

### Pass扩展

1. 自定义Pass：用户需继承[PassBase](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_registry.html#passbase)，重载Execute接口函数[Execute](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_registry.html#execute)。

2. Pass注册：调用Pass的注册接口[REG_PASS](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_registry.html#reg-pass)，把用户自己实现的Pass类注册进MindSpore Lite里。

### 算子InferShape扩展

在离线转换阶段，我们会对模型的每一个节点的输出张量进行推断，包括输出张量的Format、DataType以及Shape，因此，离线转换阶段，用户需提供自己实现的算子的推断过程，这里用户可以参考[算子Infershape扩展](https://www.mindspore.cn/lite/docs/zh-CN/master/use/runtime_cpp.html#id19)说明。

### 示例演示

#### 编译

- 环境要求

    - 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- 编译构建

  在`mindspore/lite/examples/converter_extend`目录下执行[build.sh](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/examples/converter_extend/build.sh)，将自动下载MindSpore Lite发布件并编译Demo。

  ```bash
  bash build.sh
  ```

  > 若使用该build脚本下载MindSpore Lite发布件失败，请手动下载硬件平台为CPU、操作系统为Ubuntu-x64的MindSpore Lite发布件[mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)，将解压后`tools/converter/lib`目录、`tools/converter/include`目录拷贝到`mindspore/lite/examples/converter_extend`目录下。
  >
  > 通过手动下载并且将文件放到指定位置后，需要再次执行build.sh脚本才能完成编译构建。

- 编译输出

  在`mindspore/lite/examples/converter_extend/build`目录下生成了`libconverter_extend_tutorial.so`的动态库。

#### 执行程序

1. 拷贝动态库

   将生成的`libconverter_extend_tutorial.so`动态库文件拷贝到发布件的`tools/converter/lib`下。

2. 进入发布件的转换目录

   ```bash
   cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
   ```

3. 创建converter的配置文件（converter.cfg)，文件内容如下：

   ```text
   [registry]
   plugin_path=libconverter_extend_tutorial.so      # 用户请配置动态库的正确路径
   ```

4. 将转换工具需要的动态链接库加入环境变量`LD_LIBRARY_PATH`

   ```bash
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/tools/converter/lib
   ```

5. 执行converter

   ```bash
   ./converter_lite --fmk=TFLITE --modelFile=add.tflite --configFile=converter.cfg --outputFile=add_extend
   ```

执行完后，将生成名为`add_extend.ms`的模型文件,文件路径由参数`outputFile`决定。
