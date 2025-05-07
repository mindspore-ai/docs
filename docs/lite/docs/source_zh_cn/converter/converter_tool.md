# 端侧模型转换

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/converter/converter_tool.md)

## 概述

MindSpore Lite提供离线转换模型功能的工具，支持多种类型的模型转换，转换后的模型可用于推理。命令行参数包含多种个性化选项，为用户提供方便的转换途径。

目前支持的输入模型类型有：MindSpore、TensorFlow Lite、Caffe、TensorFlow、ONNX和PyTorch。

通过转换工具转换成的`ms`模型，支持转换工具配套及更高版本的Runtime推理框架执行推理。

## Linux环境使用说明

### 环境准备

使用MindSpore Lite模型转换工具，需要进行如下环境准备工作。

- [编译](https://www.mindspore.cn/lite/docs/zh-CN/master/build/build.html)或[下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)模型转换工具。
- 将转换工具需要的动态链接库加入环境变量LD_LIBRARY_PATH。

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ```

    ${PACKAGE_ROOT_PATH}是编译或下载得到的包解压后的路径。
- 编译MindSpore Lite包时若使用的是Python3.11，则使用转换工具以及推理工具时需要将使用的Python动态链接库加入环境变量LD_LIBRARY_PATH。

    ```bash
    export LD_LIBRARY_PATH=${PATHON_ROOT_PATH}/lib:${LD_LIBRARY_PATH}
    ```

    ${PATHON_ROOT_PATH}为使用的Python环境所在路径。待解耦Python依赖后该环境变量无需设置。

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
            ├── libmindspore_glog.so.0         # Glog的动态库
            ├── libmslite_converter_plugin.so  # 注册插件的动态库
            ├── libopencv_core.so.4.5          # OpenCV的动态库
            ├── libopencv_imgcodecs.so.4.5     # OpenCV的动态库
            └── libopencv_imgproc.so.4.5       # OpenCV的动态库
```

### 参数说明

MindSpore Lite模型转换工具提供了多种参数设置，用户可根据需要来选择使用。此外，用户可输入`./converter_lite --help`获取实时帮助。

下面提供详细的参数说明。

| 参数  |  是否必选   |  参数说明  | 取值范围 | 默认值 | 备注 |
| -------- | ------- | ----- | --- | ---- | ---- |
| `--help` | 否 | 打印全部帮助信息。 | - | - | - |
| `--fmk=<FMK>`  | 是 | 输入模型的原始格式。 | MINDIR、CAFFE、TFLITE、TF、ONNX、PYTORCH、MSLITE | - | 只有在Micro代码生成时，才支持设置为MSLITE |
| `--modelFile=<MODELFILE>` | 是 | 输入模型的路径。 | - | - | - |
| `--outputFile=<OUTPUTFILE>` | 是 | 输出模型的路径，不需加后缀，可自动生成`.ms`后缀。 | - | - | - |
| `--weightFile=<WEIGHTFILE>` | 转换Caffe模型时必选 | 输入模型weight文件的路径。 | - | - | - |
| `--configFile=<CONFIGFILE>` | 否 | 1）可作为训练后量化配置文件路径；2）可作为扩展功能配置文件路径。  | - | - | - |
| `--fp16=<FP16>` | 否 | 设定在模型序列化时是否需要将float32数据格式的权重存储为float16数据格式。 | on、off | off | - |
| `--inputShape=<INPUTSHAPE>` | 否 | 设定模型输入的维度，输入维度的顺序和原始模型保持一致。对某些特定的模型可以进一步优化模型结构，但是转化后的模型将可能失去动态shape的特性。多个输入用`;`分割，同时加上双引号`""`。 | e.g.  "inTensorName_1: 1,32,32,4;inTensorName_2:1,64,64,4;" | - | - |
| `--saveType=<SAVETYPE>` | 否 | 设定导出的模型为`mindir`模型或者`ms`模型。 | MINDIR、MINDIR_LITE | MINDIR_LITE | 端侧推理版本只有设置为MINDIR_LITE转出的模型才可以推理 |
| `--optimize=<OPTIMIZE>` | 否 | 设定转换模型的过程所完成的优化。 | none、general、gpu_oriented、ascend_oriented | general | - | - |
| `--inputDataFormat=<INPUTDATAFORMAT>` | 否 | 设定导出模型的输入format，只对四维输入有效。 | NHWC、NCHW | NHWC | - |
| `--decryptKey=<DECRYPTKEY>` | 否 | 设定用于加载密文MindIR时的密钥，密钥用十六进制表示，只对`fmk`为MINDIR时有效。 | - | - | - |
| `--decryptMode=<DECRYPTMODE>` | 否 | 设定加载密文MindIR的模式，只在指定了decryptKey时有效。 | AES-GCM、AES-CBC | AES-GCM | - |
| `--inputDataType=<INPUTDATATYPE>` | 否 | 设定量化模型输入tensor的data type。仅当模型输入tensor的量化参数（scale和zero point）齐备时有效。默认与原始模型输入tensor的data type保持一致。 | FLOAT32、INT8、UINT8、DEFAULT | DEFAULT | - |
| `--outputDataType=<OUTPUTDATATYPE>` | 否 | 设定量化模型输出tensor的data type。仅当模型输出tensor的量化参数（scale和zero point）齐备时有效。默认与原始模型输出tensor的data type保持一致。 | FLOAT32、INT8、UINT8、DEFAULT | DEFAULT | - |
| `--outputDataFormat=<OUTPUTDATAFORMAT>` | 否 | 设定导出模型的输出format，只对四维输出有效。 | NHWC、NCHW | - | - |
| `--encryptKey=<ENCRYPTKEY>` | 否 | 设定导出加密`ms`模型的密钥，密钥用十六进制表示。仅支持 AES-GCM，密钥长度仅支持16Byte。 | - | - | - |
| `--encryption=<ENCRYPTION>` | 否 | 设定导出`ms`模型时是否加密，导出加密可保护模型完整性，但会增加运行时初始化时间。 | true、false | false | - |
| `--infer=<INFER>` | 否 | 设定是否在转换完成时进行预推理。 | true、false | false | - |

> - 参数名和参数值之间用等号连接，中间不能有空格。
> - 由于支持转换PyTorch模型的编译选项默认关闭，因此下载的安装包不支持转换PyTorch模型。需要打开指定编译选项进行本地编译。转换PyTorch模型需满足以下前提：编译前需要`export MSLITE_ENABLE_CONVERT_PYTORCH_MODEL=on && export LIB_TORCH_PATH="/home/user/libtorch"`，用户可以下载[CPU版本libtorch](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip)后解压到`/home/user/libtorch`的目录下。转换前加入libtorch的环境变量：`export LD_LIBRARY_PATH="/home/user/libtorch/lib:${LD_LIBRARY_PATH}"`。
> - Caffe模型一般分为两个文件：`*.prototxt`模型结构，对应`--modelFile`参数；`*.caffemodel`模型权值，对应`--weightFile`参数。
> - `--fp16`的优先级很低，如果开启了量化，那么对于已经量化的权重，`--fp16`不会再次生效。总而言之，该选项只会在序列化时对模型中的float32的权重生效。
> - `inputDataFormat`：一般在集成NCHW规格的三方硬件场景下，设为NCHW比NHWC会有较明显的性能提升。在其他场景下，用户也可按需设置。
> - `configFile`配置文件采用`key=value`的方式定义相关参数，量化相关的配置参数详见[量化](https://www.mindspore.cn/lite/docs/zh-CN/master/advanced/quantization.html)，扩展功能相关的配置参数详见[扩展配置](https://www.mindspore.cn/lite/docs/zh-CN/master/advanced/third_party/converter_register.html#扩展配置)。
> - `--optimize`该参数是用来设定在离线转换的过程中需要完成哪些特定的优化。如果该参数设置为none，那么在模型的离线转换阶段将不进行相关的图优化操作，相关的图优化操作将会在执行推理阶段完成。该参数的优点在于转换出来的模型由于没有经过特定的优化，可以直接部署到CPU/GPU/Ascend任意硬件后端；而带来的缺点是推理执行时模型的初始化时间增长。如果设置成general，表示离线转换过程会完成通用优化，包括常量折叠，算子融合等（转换出的模型只支持CPU/GPU后端，不支持Ascend后端）。如果设置成gpu_oriented，表示转换过程中会完成通用优化和针对GPU后端的额外优化（转换出来的模型只支持GPU后端）。如果设置成ascend_oriented，表示转换过程中只完成针对Ascend后端的优化（转换出来的模型只支持Ascend后端）。
> - 加解密功能仅在[编译](https://www.mindspore.cn/lite/docs/zh-CN/master/build/build.html)时设置为`MSLITE_ENABLE_MODEL_ENCRYPTION=on`时生效，并且仅支持Linux x86平台。其中密钥为十六进制表示的字符串，如密钥定义为`b'0123456789ABCDEF'`对应的十六进制表示为`30313233343536373839414243444546`，Linux平台用户可以使用`xxd`工具对字节表示的密钥进行十六进制表达转换。
    需要注意的是，加解密算法在1.7版本进行了更新，导致新版的converter工具不支持对1.6及其之前版本的MindSpore加密导出的模型进行转换。
> - `--input_shape`参数以及dynamicDims参数在转换时会被存入模型中，在使用模型时可以调用model.get_model_info("input_shape")以及model.get_model_info("dynamic_dims")来获取。

### CPU模型编译优化

如果转换后的ms模型使用Android CPU后端进行推理，并且对模型编译阶段时延要求较高。可以尝试开启此优化，在`configFile`配置文件中增加配置项`[cpu_option_cfg_param]`，得到编译更高效的模型。目前仅对模型中含有Matmul算子并且数据类型为`float32`或开启动态量化时有优化效果。

| 参数 | 属性 | 功能描述 | 取值范围 |
|--------|--------|--------|--------|
|    `architecture`    |    必选    |     目标架构，当前仅支持ARM64    |     ARM64    |
|    `instruction`    |    必选    |    目标指令集，当前仅支持SMID_DOT    |    SIMD_DOT    |

### 使用示例

下面选取了几个常用示例，说明转换命令的使用方法。

- 以Caffe模型LeNet为例，执行转换命令。

   ```bash
   ./converter_lite --fmk=CAFFE --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
   ```

   本例中，因为采用了Caffe模型，所以需要模型结构、模型权值两个输入文件。再加上其他必需的fmk类型和输出路径两个参数，即可成功执行。

   结果显示为：

   ```text
   CONVERT RESULT SUCCESS:0
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

    - PyTorch模型`model.pt`

        ```bash
        export LD_LIBRARY_PATH="/home/user/libtorch/lib:${LD_LIBRARY_PATH}"
        export LIB_TORCH_PATH="/home/user/libtorch"
        ./converter_lite --fmk=PYTORCH --modelFile=model.pt --outputFile=model
        ```

    - PyTorch模型`model.pth`

        ```bash
        export LD_LIBRARY_PATH="/home/user/libtorch/lib:${LD_LIBRARY_PATH}"
        export LIB_TORCH_PATH="/home/user/libtorch"
        ./converter_lite --fmk=PYTORCH --modelFile=model.pth --outputFile=model
        ```

        > 为了转换PyTorch模型，以下前提必须满足：编译前需要`export MSLITE_ENABLE_CONVERT_PYTORCH_MODEL=on && export LIB_TORCH_PATH="/home/user/libtorch"`，用户可以下载[CPU版本libtorch](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip)后解压到`/home/user/libtorch`路径。转换前加入libtorch的环境变量，`export LD_LIBRARY_PATH="/home/user/libtorch/lib:${LD_LIBRARY_PATH}"`。

    以上几种情况下，均显示如下转换成功提示，且同时获得`model.ms`目标文件。

    ```text
    CONVERT RESULT SUCCESS:0
    ```

> 量化示例请参考<https://www.mindspore.cn/lite/docs/zh-CN/master/advanced/quantization.html>。

## Windows环境使用说明

### 环境准备

使用MindSpore Lite模型转换工具，需要进行如下环境准备工作。

- [编译](https://www.mindspore.cn/lite/docs/zh-CN/master/build/build.html)或[下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)模型转换工具。
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
            ├── libmindspore_glog.dll            # Glog的动态库
            ├── libmslite_converter_plugin.dll   # 注册插件的动态库
            ├── libmslite_converter_plugin.dll.a # 注册插件的动态库的链接文件
            ├── libssp-0.dll          # MinGW动态库
            ├── libstdc++-6.dll       # MinGW动态库
            └── libwinpthread-1.dll   # MinGW动态库
```

### 参数说明

参考Linux环境模型转换工具的[参数说明](https://www.mindspore.cn/lite/docs/zh-CN/master/converter/converter_tool.html#参数说明)。

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
   CONVERT RESULT SUCCESS:0
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
    CONVERT RESULT SUCCESS:0
    ```
