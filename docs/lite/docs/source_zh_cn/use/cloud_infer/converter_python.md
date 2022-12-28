# 使用Python接口模型转换

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/use/cloud_infer/converter_python.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

MindSpore Lite云侧推理支持通过Python接口进行模型转换，支持多种类型的模型转换，转换后的mindir模型可用于推理。接口包含多种个性化参数，为用户提供方便的转换途径。本教程介绍如何使用[Python接口](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Converter.html)进行模型转换。

目前支持的输入模型类型有：MindSpore、TensorFlow Lite、Caffe、TensorFlow、ONNX。

当输入模型类型是MindSpore时，由于已经是`mindir`模型，建议两种做法：

1. 不需要经过离线转换，直接进行推理执行。

2. 设置参数no_fusion的值为False，在离线阶段完成相关优化，减少推理执行的初始化时间。

## Linux环境使用说明

### 环境准备

使用MindSpore Lite云侧推理的Python接口进行模型转换，需要进行如下环境准备工作。

- [编译](https://www.mindspore.cn/lite/docs/zh-CN/master/use/build.html)或[下载](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)含Converter组件的MindSpore Lite云侧推理的Whl安装包。

    > 当前，提供下载Python3.7版本对应的安装包，若需要其他Python版本，请使用编译功能生成安装包。

- 然后使用`pip install`命令进行安装。安装后可以使用以下命令检查是否安装成功：若无报错，则表示安装成功。

    ```bash
    python -c "import mindspore_lite"
    ```

### 目录结构

安装成功后，可使用`pip show mindspore_lite`命令查看MindSpore Lite云侧推理的Python模块的安装位置。

```text
mindspore_lite
├── __pycache__
├── include
├── lib
│   ├── _c_lite_wrapper.cpython-37m-x86_64-linux-gnu.so         # MindSpore Lite 云侧推理python模块封装C++接口的框架的动态库
│   ├── libmindspore_converter.so                               # 模型转换动态库
│   ├── libmindspore_core.so                                    # MindSpore Core动态库
│   ├── libmindspore_glog.so.0                                  # Glog的动态库
│   ├── libmindspore-lite.so                                    # MindSpore Lite云侧推理的动态库
│   ├── libmslite_converter_plugin.so                           # 模型转换插件
│   ├── libascend_pass_plugin.so                                # 注册昇腾后端图优化插件动态库
│   ├── libmindspore_shared_lib.so                              # 适配昇腾后端的动态库
│   ├── libascend_kernel_plugin.so                              # 昇腾后端kernel插件
│   ├── libtensorrt_plugin.so                                   # tensorrt后端kernel插件
│   ├── libopencv_core.so.4.5                                   # OpenCV的动态库
│   ├── libopencv_imgcodecs.so.4.5                              # OpenCV的动态库
│   └── libopencv_imgproc.so.4.5                                # OpenCV的动态库
├── __init__.py        # 初始化包
├── context.py         # context接口相关代码
├── converter.py       # converter接口相关代码，转换入口
├── model.py           # model接口相关代码，推理入口
├── tensor.py          # tensor接口相关代码
└── version.py         # MindSpore Lite云侧推理版本号
```

### 参数说明

MindSpore Lite云侧推理的Python接口模型转换提供了多种参数设置，用户可根据需要来选择使用。

使用场景：将第三方模型转换生成MindSpore模型。

下面提供详细的参数说明以及与[推理模型离线转换](https://www.mindspore.cn/lite/docs/zh-CN/master/use/cloud_infer/converter_tool.html)中参数的对应关系。

| Python接口模型转换参数 | 参数类型  | 对应模型离线转换的参数  |  是否必选   |  参数说明  | 取值范围 | 默认值 | 备注 |
| -------- | ----- | -------- | ------- | ----- | --- | ---- | ---- |
| fmk_type | FmkType | `--fmk=<FMK>`  | 是 | 输入模型框架类型。 | FmkType.TF、FmkType.CAFFE、FmkType.ONNX、FmkType.TFLITE | - |
| model_file | str | `--modelFile=<MODELFILE>` | 是 | 转换时的输入模型文件路径。 | - | - |
| output_file | str | `--outputFile=<OUTPUTFILE>` | 是 | 转换时的输出模型的路径，可自动生成`.mindir`后缀。 | - | - |
| weight_file | str | `--weightFile=<WEIGHTFILE>` | 转换Caffe模型时必选 | 输入模型权重文件路径。 | - | "" |
| config_file | str | `--configFile=<CONFIGFILE>` | 否 | Converter的配置文件路径，可配置训练后量化或离线拆分算子并行或禁用算子融合功能并将插件设置为so路径等功能。 | - | "" |
| input_shape | dict{string:list\[int]} | `--inputShape=<INPUTSHAPE>` | 否 | 设置模型输入的维度，输入维度的顺序和原始模型保持一致。如：{"inTensor1": \[1, 32, 32, 32], "inTensor2": \[1, 1, 32, 32]} | - | None，None等同于{} |
| export_mindir | ModelType | `--exportMindIR=<EXPORTMINDIR>` | 是 | 设置导出模型文件的类型。| ModelType.MINDIR、ModelType.MINDIR_LITE | ModelType.MINDIR | MINDIR模型使用MindSpore Lite云侧推理安装包，MINDIR_LITE使用MindSpore Lite端侧推理安装包|
| no_fusion | bool | `--NoFusion=<NOFUSION>` | 否 | 是否避免融合优化，默认允许融合优化。 | True、False | False |
| weight_fp16 | bool | `--fp16=<FP16>` | 否 | 设置在模型序列化时是否需要将Float32数据格式的权重存储为Float16数据格式。 | True、False | False | 暂不支持 |
| input_format | Format | `--inputDataFormat=<INPUTDATAFORMAT>` | 否 | 设置导出模型的输入format，只对4维输入有效。 | Format.NCHW、Format.NHWC | Format.NHWC | 暂不支持 |
| input_data_type | DataType | `--inputDataType=<INPUTDATATYPE>` | 否 | 设置量化模型输入Tensor的data type。仅当模型输入Tensor的量化参数（`scale`和`zero point`）都具备时有效。默认与原始模型输入Tensor的data type保持一致。 | DataType.FLOAT32、DataType.INT8、DataType.UINT8、DataType.UNKNOWN | DataType.FLOAT32 | 暂不支持 |
| output_data_type | DataType | `--outputDataType=<OUTPUTDATATYPE>` | 否 | 设置量化模型输出Tensor的data type。仅当模型输出Tensor的量化参数（`scale`和`zero point`）都具备时有效。默认与原始模型输出Tensor的data type保持一致。 | DataType.FLOAT32、DataType.INT8、DataType.UINT8、DataType.UNKNOWN | DataType.FLOAT32 | 暂不支持 |
| decrypt_key | str | `--decryptKey=<DECRYPTKEY>` | 否 | 设置用于加载密文MindIR时的密钥，密钥用十六进制表示，只对`fmk_type`为MINDIR时有效。 | - | "" | 暂不支持 |
| decrypt_mode | str | `--decryptMode=<DECRYPTMODE>` | 否 | 设置加载密文MindIR的模式，只在指定了`decrypt_key`时有效。 | "AES-GCM"、"AES-CBC" | "AES-GCM" | 暂不支持 |
| enable_encryption | bool | `--encryption=<ENCRYPTION>` | 否 | 导出模型时是否加密，导出加密可保护模型完整性，但会增加运行时初始化时间。 | True、False | False | 暂不支持 |
| encrypt_key | str | `--encryptKey=<ENCRYPTKEY>` | 否 | 设置用于加密文件的密钥，以十六进制字符表示。仅支持当`decrypt_mode`是"AES-GCM"，密钥长度为16。 | - | "" | 暂不支持 |
| infer | bool | `--infer=<INFER>` | 否 | 是否在转换完成时进行预推理。 | True、False | False | 暂不支持 |
| train_model | bool | `--trainModel=<TRAINMODEL>` | 否 | 模型是否将在设备上进行训练。 | True、False | False | 暂不支持 |

> `fmk_type`参数有关详细信息，请参见[FmkType](https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.FmkType.html)
>
> `model_file`举例："/home/user/model.prototxt"。不同类型应模型后缀举例：TF: "model.pb" | CAFFE: "model.prototxt" | ONNX: "model.onnx" | TFLITE: "model.tflite"。
>
> Caffe模型一般分为两个文件：`*.prototxt`是模型结构，对应`model_file`参数；`model.caffemodel`是模型权值，对应`weight_file`参数。
>
> `config_file`配置文件采用`key = value`的方式定义相关参数。
>
> `input_shape`在以下场景下，用户可能需要设置该参数：
>
> - 用法1：待转换模型的输入是动态shape，准备采用固定shape推理，则设置该参数为固定shape。设置之后，在对Converter后的模型进行推理时，默认输入的shape与该参数设置一样，无需再进行resize操作。
> - 用法2：无论待转换模型的原始输入是否为动态shape，准备采用固定shape推理，并希望模型的性能尽可能优化，则设置该参数为固定shape。设置之后，将对模型结构进一步优化，但转换后的模型可能会失去动态shape的特征（部分跟shape强相关的算子会被融合）。
>

### 使用示例

下面选取了几个常用示例，说明转换命令的使用方法。

- 以Caffe模型LeNet为例。

    ```python
    import mindspore_lite as mslite
    converter = mslite.Converter(fmk_type=mslite.FmkType.CAFFE, model_file="lenet.prototxt",output_file="lenet", weight_file="lenet.caffemodel",export_mindir=mslite.ModelType.MINDIR)
    converter.converter()
    ```

    本例中，因为采用了Caffe模型，所以需要模型结构、模型权值两个输入文件。再加上其他必需的fmk类型和输出路径两个参数，即可成功执行。

    结果显示为：

    ```text
    CONVERT RESULT SUCCESS:0
    ```

    这表示已经成功将Caffe模型转化为MindSpore Lite云侧推理模型，获得新文件`lenet.mindir`。

- 以MindSpore、TensorFlow Lite、TensorFlow和ONNX模型为例，执行转换命令。

    - MindSpore模型`model.mindir`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter(fmk_type=mslite.FmkType.MINDIR, model_file="model.mindir",output_file="model",export_mindir=mslite.ModelType.MINDIR, no_fusion=False)
        converter.converter()
        ```

    - TensorFlow Lite模型`model.tflite`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="model.tflite",output_file="model",export_mindir=mslite.ModelType.MINDIR)
        converter.converter()
        ```

    - TensorFlow模型`model.pb`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter(fmk_type=mslite.FmkType.TF, model_file="model.pb", output_file="model",export_mindir=mslite.ModelType.MINDIR)
        converter.converter()
        ```

    - ONNX模型`model.onnx`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter(fmk_type=mslite.FmkType.ONNX, model_file="model.onnx", output_file="model",export_mindir=mslite.ModelType.MINDIR)
        converter.converter()
        ```

    以上几种情况下，均显示如下转换成功提示，且同时获得`model.mindir`目标文件。

    ```text
    CONVERTER RESULT SUCCESS:0
    ```
