# 推理模型转换

`Windows` `Linux` `模型转换` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/docs/source_zh_cn/use/converter_tool.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## 概述

MindSpore Lite提供离线转换模型功能的工具，支持多种类型的模型转换，转换后的模型可用于推理。命令行参数包含多种个性化选项，为用户提供方便的转换途径。

目前支持的输入格式有：MindSpore、TensorFlow Lite、Caffe、TensorFlow和ONNX。

通过转换工具转换成的`ms`模型，支持转换工具配套及更高版本的Runtime推理框架执行推理。

## Linux环境使用说明

### 环境准备

使用MindSpore Lite模型转换工具，需要进行如下环境准备工作。

- [编译](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/build.html)或[下载](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/downloads.html)模型转换工具。
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
        ├── converter                # 模型转换工具
        │   └── converter_lite       # 可执行程序
        └── lib                      # 转换工具依赖的动态库
            └── libglog.so.0         # Glog的动态库
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
| `--quantType=<QUANTTYPE>` | 否 | 设置模型的量化类型。 | WeightQuant：训练后量化（权重量化）<br>PostTraining：训练后量化（全量化） | - |
| `--bitNum=<BITNUM>` | 否 | 设定训练后量化（权重量化）的比特数，目前支持1bit～16bit量化 | \[1，16] | 8 |
| `--quantWeightSize=<QUANTWEIGHTSIZE>` | 否 | 设定参与训练后量化（权重量化）的卷积核尺寸阈值，若卷积核尺寸大于该值，则对此权重进行量化 |  \[0，+∞） | 0 |
| `--quantWeightChannel=<QUANTWEIGHTCHANNEL>` | 否 | 设定参与训练后量化（权重量化）的卷积通道数阈值，若卷积通道数大于该值，则对此权重进行量化 | \[0，+∞） | 16 |
| `--configFile=<CONFIGFILE>` | 否 | 1）可作为训练后量化（全量化）校准数据集配置文件路径；2）可作为转换器的配置文件路径。  |  - | -  |
| `--fp16=<FP16>` | 否 | 设定在模型序列化时是否需要将Float32数据格式的权重存储为Float16数据格式. | on、off | off |
| `--inputShape=<INPUTSHAPE>` | 否 | 设定模型输入的维度，默认与原始模型的输入一致。对某些特定的模型可以进一步常量折叠，比如存在shape算子的模型，但是转化后的模型将失去动态shape的特性。e.g.  inTensorName: 1,32,32,4 | -| - |

> - 参数名和参数值之间用等号连接，中间不能有空格。
> - Caffe模型一般分为两个文件：`*.prototxt`模型结构，对应`--modelFile`参数；`*.caffemodel`模型权值，对应`--weightFile`参数。
> - 为保证权重量化的精度，建议`--bitNum`参数设定范围为8bit～16bit。
> - 全量化目前仅支持激活值8bit、权重8bit的量化方式。
> - `--fp16`的优先级很低，比如如果开启了量化，那么对于已经量化的权重，`--fp16`不会再次生效。总而言之，该选项只会在序列化时对模型中的Float32的权重生效。

`configFile`配置文件采用`key=value`的方式定义相关参数，可配置的`key`如下:

|   参数名  |  属性   |     功能描述    |  参数类型 |   默认值 | 取值范围  |
| -------- | ------- | -----          | -----    | -----     |  ----- |
| image_path  | 必选 | 存放校准数据集的目录；如果模型有多个输入，请依次填写对应的数据所在目录，目录路径间请用`,`隔开 |      String                |   -   | 该目录存放可直接用于执行推理的输入数据。由于目前框架还不支持数据预处理，所有数据必须事先完成所需的转换，使得它们满足推理的输入要求 |
| batch_count | 可选 | 使用的输入数目       | Integer  |  100  | （0，+∞） |
| method_x | 可选 | 网络层输入输出数据量化算法 | String  |  KL  | KL、MAX_MIN、RemovalOutlier。 <br> KL：基于[KL散度](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)对数据范围作量化校准。 <br> MAX_MIN：基于最大值、最小值计算数据的量化参数。 <br> RemovalOutlier：按照一定比例剔除数据的极大极小值，再计算量化参数。 <br> 在校准数据集与实际推理时的输入数据相吻合的情况下，推荐使用MAX_MIN；而在校准数据集噪声比较大的情况下，推荐使用KL或者RemovalOutlier      |
| thread_num | 可选 | 使用校准数据集执行推理流程时的线程数 | Integer  |  1  |  （0，+∞）   |
| bias_correction | 可选 | 是否对量化误差进行校正 | Boolean  |  false  |  true、flase。使能后，能提升转换后的模型精度，建议设置为true |
| plugin_path | 可选 | 第三方库加载路径  | String  |  -  | 如有多个请用`;`分隔   |
| disable_fusion | 可选 | 是否关闭融合优化 | String  |  off  |  off、on |

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

- 如果转换命令执行失败，程序会返回一个[错误码](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/errorcode_and_metatype.html)。

> 训练后量化示例请参考<https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/post_training_quantization.html>。

## Windows环境使用说明

### 环境准备

使用MindSpore Lite模型转换工具，需要进行如下环境准备工作。

- [编译](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/build.html)或[下载](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/downloads.html)模型转换工具。
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
        ├── include
        ├── converter
        │   └── converter_lite.exe    # 可执行程序
        └── lib
            ├── libgcc_s_seh-1.dll    # MinGW动态库
            ├── libglog.dll           # Glog的动态库
            ├── libssp-0.dll          # MinGW动态库
            ├── libstdc++-6.dll       # MinGW动态库
            └── libwinpthread-1.dll   # MinGW动态库
```

### 参数说明

参考Linux环境模型转换工具的[参数说明](https://www.mindspore.cn/lite/docs/zh-CN/r1.3/use/converter_tool.html#id3)。

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

- 如果转换命令执行失败，程序会返回一个[错误码](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/errorcode_and_metatype.html)。
