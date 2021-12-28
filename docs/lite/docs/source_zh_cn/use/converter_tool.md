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
| `--device=<DEVICE>` | 否 | 指定转换模型运行的设备类型 | 仅支持Ascend310 | DEFAULT |

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
