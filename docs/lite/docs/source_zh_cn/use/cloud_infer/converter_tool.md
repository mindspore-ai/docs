# 推理模型离线转换

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/use/cloud_infer/converter_tool.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore Lite云侧推理提供离线转换模型功能的工具，支持多种类型的模型转换，转换后的模型可用于推理。命令行参数包含多种个性化选项，为用户提供方便的转换途径。

目前支持的输入格式有：MindSpore、TensorFlow Lite、Caffe、TensorFlow和ONNX。

通过转换工具转换成的`mindir`模型，支持转换工具配套及更高版本的Runtime推理框架执行推理。

## Linux环境使用说明

### 环境准备

使用MindSpore Lite云侧推理模型转换工具，需要进行如下环境准备工作。

- [编译](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/build.html)或[下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)模型转换工具。
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
        ├── include                            # 自定义算子、模型解析、节点解析、转换优化注册头文件
        ├── converter                          # 模型转换工具
        │   └── converter_lite                 # 可执行程序
        └── lib                                # 转换工具依赖的动态库
            ├── libmindspore_glog.so.0         # Glog动态库
            ├── libascend_pass_plugin.so       # 注册昇腾后端图优化插件动态库
            ├── libmindspore_shared_lib.so     # 适配昇腾后端的动态库
            ├── libmindspore_converter.so      # 模型转换动态库
            ├── libmslite_converter_plugin.so  # 模型转换插件
            ├── libmindspore_core.so           # MindSpore Core动态库
            ├── libopencv_core.so.4.5          # OpenCV的动态库
            ├── libopencv_imgcodecs.so.4.5     # OpenCV的动态库
            └── libopencv_imgproc.so.4.5       # OpenCV的动态库
        ├── third_party                        # 第三方模型proto定义  
```

### 参数说明

MindSpore Lite云侧推理模型转换工具提供了多种参数设置，用户可根据需要来选择使用。此外，用户可输入`./converter_lite --help`获取实时帮助。

下面提供详细的参数说明。

| 参数  |  是否必选   |  参数说明  | 取值范围 | 默认值 | 备注 |
| -------- | ------- | ----- | --- | ---- | ---- |
| `--help` | 否 | 打印全部帮助信息。 | - | - |
| `--fmk=<FMK>`  | 是 | 输入模型的原始格式。 | MINDIR、CAFFE、TFLITE、TF、ONNX | - | - |
| `--modelFile=<MODELFILE>` | 是 | 输入模型的路径。 | - | - | - |
| `--outputFile=<OUTPUTFILE>` | 是 | 输出模型的路径，不需加后缀，可自动生成`.mindir`后缀。 | - | - | - |
| `--weightFile=<WEIGHTFILE>` | 转换Caffe模型时必选 | 输入模型weight文件的路径。 | - | - | - |
| `--configFile=<CONFIGFILE>` | 否 | 1）可作为训练后量化配置文件路径；2）可作为扩展功能配置文件路径。  | - | - | - |
| `--inputShape=<INPUTSHAPE>` | 否 | 设定模型输入的维度，输入维度的顺序和原始模型保持一致。对某些特定的模型可以进一步优化模型结构，但是转化后的模型将可能失去动态shape的特性。多个输入用`;`分割，同时加上双引号`""`。 | e.g.  "inTensorName_1: 1,32,32,4;inTensorName_2:1,64,64,4;" | - | - |
| `--saveType=<SAVETYPE>` | 是 | 设定导出的模型为`mindir`模型或者`ms`模型。 | MINDIR、MINDIR_LITE | MINDIR | 云侧推理版本只有设置为MINDIR转出的模型才可以推理 |
| `--optimize=<OPTIMIZE>` | 否 | 设定转换模型的过程所完成的优化。 | none、general、ascend_oriented | general | 当saveType设置为MINDIR的情况下，该参数的默认值为none。 |
| `--fp16=<FP16>` | 否 | 设定在模型序列化时是否需要将Float32数据格式的权重存储为Float16数据格式。 | on、off | off | 暂不支持 |
| `--decryptKey=<DECRYPTKEY>` | 否 | 设定用于加载密文MindIR时的密钥，密钥用十六进制表示，只对`fmk`为MINDIR时有效。 | - | - | 暂不支持 |
| `--decryptMode=<DECRYPTMODE>` | 否 | 设定加载密文MindIR的模式，只在指定了decryptKey时有效。 | AES-GCM、AES-CBC | AES-GCM | 暂不支持 |
| `--inputDataType=<INPUTDATATYPE>` | 否 | 设定量化模型输入tensor的data type。仅当模型输入tensor的量化参数（scale和zero point）齐备时有效。默认与原始模型输入tensor的data type保持一致。 | FLOAT32、INT8、UINT8、DEFAULT | DEFAULT | 暂不支持 |
| `--outputDataType=<OUTPUTDATATYPE>` | 否 | 设定量化模型输出tensor的data type。仅当模型输出tensor的量化参数（scale和zero point）齐备时有效。默认与原始模型输出tensor的data type保持一致。 | FLOAT32、INT8、UINT8、DEFAULT | DEFAULT | 暂不支持 |
| `--encryptKey=<ENCRYPTKEY>` | 否 | 设定导出加密`mindir`模型的密钥，密钥用十六进制表示。仅支持 AES-GCM，密钥长度仅支持16Byte。 | - | - | 暂不支持 |
| `--encryption=<ENCRYPTION>` | 否 | 设定导出`mindir`模型时是否加密，导出加密可保护模型完整性，但会增加运行时初始化时间。 | true、false | true | 暂不支持 |
| `--infer=<INFER>` | 否 | 设定是否在转换完成时进行预推理。 | true、false | false | 暂不支持 |
| `--inputDataFormat=<INPUTDATAFORMAT>` | 否 | 设定导出模型的输入format，只对四维输入有效。 | NHWC、NCHW | NHWC | 暂不支持 |

注意事项：

- 参数名和参数值之间用等号连接，中间不能有空格。
- Caffe模型一般分为两个文件：`*.prototxt`模型结构，对应`--modelFile`参数；`*.caffemodel`模型权值，对应`--weightFile`参数。
- `configFile`配置文件采用`key=value`的方式定义相关参数。
- `--optimize`该参数是用来设定在离线转换的过程中需要完成哪些特定的优化。如果该参数设置为none，那么在模型的离线转换阶段将不进行相关的图优化操作，相关的图优化操作将会在执行推理阶段完成。该参数的优点在于转换出来的模型由于没有经过特定的优化，可以直接部署到CPU/GPU/Ascend任意硬件后端；而带来的缺点是推理执行时模型的初始化时间增长。如果设置成general，表示离线转换过程会完成通用优化，包括常量折叠，算子融合等（转换出的模型只支持CPU/GPU后端，不支持Ascend后端）。如果设置成ascend_oriented，表示转换过程中只完成针对Ascend后端的优化（转换出来的模型只支持Ascend后端）。
- 针对MindSpore模型，由于已经是`mindir`模型，建议两种做法：

    不需要经过离线转换，直接进行推理执行。

    使用离线转换并设置--optimize为general，在离线阶段完成相关优化，减少推理执行的初始化时间。

### 使用示例

下面选取了几个常用示例，说明转换命令的使用方法。

- 以Caffe模型LeNet为例，执行转换命令。

    ```bash
    ./converter_lite --fmk=CAFFE --saveType=MINDIR --optimize=none --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
    ```

    本例中，因为采用了Caffe模型，所以需要模型结构、模型权值两个输入文件。再加上其他必需的fmk类型和输出路径两个参数，即可成功执行。

    结果显示为：

    ```text
    CONVERT RESULT SUCCESS:0
    ```

    这表示已经成功将Caffe模型转化为MindSpore Lite云侧推理模型，获得新文件`lenet.mindir`。

- 以MindSpore、TensorFlow Lite、TensorFlow和ONNX模型为例，执行转换命令。

    - MindSpore模型`model.mindir`

    ```bash
    ./converter_lite --fmk=MINDIR --saveType=MINDIR --optimize=general --modelFile=model.mindir --outputFile=model
    ```

    - TensorFlow Lite模型`model.tflite`

    ```bash
    ./converter_lite --fmk=TFLITE --saveType=MINDIR --optimize=none --modelFile=model.tflite --outputFile=model
    ```

    - TensorFlow模型`model.pb`

    ```bash
    ./converter_lite --fmk=TF --saveType=MINDIR --optimize=none --modelFile=model.pb --outputFile=model
    ```

    - ONNX模型`model.onnx`

    ```bash
    ./converter_lite --fmk=ONNX --saveType=MINDIR --optimize=none --modelFile=model.onnx --outputFile=model
    ```

    以上几种情况下，均显示如下转换成功提示，且同时获得`model.mindir`目标文件。

    ```text
    CONVERTER RESULT SUCCESS:0
    ```
