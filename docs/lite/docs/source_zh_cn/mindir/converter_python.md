# 使用Python接口模型转换

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/mindir/converter_python.md)

## 概述

MindSpore Lite云侧推理支持通过Python接口进行模型转换，支持多种类型的模型转换，转换后的mindir模型可用于推理。接口包含多种个性化参数，为用户提供方便的转换途径。本教程介绍如何使用[Python接口](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/mindspore_lite/mindspore_lite.Converter.html)进行模型转换。

目前支持的输入模型类型有：MindSpore、TensorFlow Lite、Caffe、TensorFlow、ONNX。

当输入模型类型是MindSpore时，由于已经是`mindir`模型，建议两种做法：

1. 不需要经过离线转换，直接进行推理执行。

2. 使用离线转换，CPU/GPU后端设置optimize为"general"（使能通用优化），GPU后端设置optimize为"gpu_oriented"（在通用优化的基础上，使能针对GPU的额外优化），NPU后端设置optimize为"ascend_oriented"，在离线阶段完成相关优化，减少推理执行的初始化时间。

## Linux环境使用说明

### 环境准备

使用MindSpore Lite云侧推理的Python接口进行模型转换，需要进行如下环境准备工作。

- [编译](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html)或[下载](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/use/downloads.html)含Converter组件的MindSpore Lite云侧推理的Whl安装包。

    > 当前，提供下载Python3.7版本对应的安装包，若需要其他Python版本，请使用编译功能生成安装包。

- 然后使用`pip install`命令进行安装。安装后可以使用以下命令检查是否安装成功：若无报错，则表示安装成功。

    ```bash
    python -c "import mindspore_lite"
    ```

- 安装后可以使用以下命令检查MindSpore Lite内置的AKG是否安装成功：若无报错，则表示安装成功。

  ```bash
  python -c "import mindspore_lite.akg"
  ```

### 目录结构

安装成功后，可使用`pip show mindspore_lite`命令查看MindSpore Lite云侧推理的Python模块的安装位置。

```text
mindspore_lite
├── __pycache__
├── akg                                                         # AKG相关的接口
├── include
├── lib
|   ├── libakg.so                                               # AKG使用的动态链接库
│   ├── _c_lite_wrapper.cpython-37m-x86_64-linux-gnu.so         # MindSpore Lite 云侧推理python模块封装C++接口的框架的动态库
│   ├── libmindspore_converter.so                               # 模型转换动态库
│   ├── libmindspore_core.so                                    # MindSpore Core动态库
│   ├── libmindspore_glog.so.0                                  # Glog的动态库
│   ├── libmindspore-lite.so                                    # MindSpore Lite云侧推理的动态库
│   ├── libmslite_converter_plugin.so                           # 模型转换插件
│   ├── libascend_pass_plugin.so                                # 注册昇腾后端图优化插件动态库
│   ├── libmslite_shared_lib.so                                 # 适配昇腾后端的动态库
│   ├── libascend_kernel_plugin.so                              # 昇腾后端kernel插件
│   ├── libtensorrt_plugin.so                                   # tensorrt后端kernel插件
│   ├── libopencv_core.so.4.5                                   # OpenCV的动态库
│   ├── libopencv_imgcodecs.so.4.5                              # OpenCV的动态库
│   └── libopencv_imgproc.so.4.5                                # OpenCV的动态库
├── __init__.py        # 初始化包
├── _checkparam.py     # 校验参数工具
├── context.py         # context接口相关代码
├── converter.py       # converter接口相关代码，转换入口
├── model.py           # model接口相关代码，推理入口
├── tensor.py          # tensor接口相关代码
└── version.py         # MindSpore Lite云侧推理版本号
```

### 属性说明

MindSpore Lite云侧推理的Python接口模型转换提供了多种属性设置，用户可根据需要来选择使用。

下面提供详细的属性说明以及与[推理模型离线转换](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool.html)中参数的对应关系。

| Converter属性 | 返回值类型  | 对应模型离线转换的参数  |  说明  | 取值范围 | 备注 |
| -------- | ----- | -------- | ----- | --- | ---- |
| decrypt_key | str | `--decryptKey=<DECRYPTKEY>` | 设置用于加载密文MindIR时的密钥，密钥用十六进制表示，只对`fmk_type`为MINDIR时有效。 | - | - |
| decrypt_mode | str | `--decryptMode=<DECRYPTMODE>` | 设置加载密文MindIR的模式，只在指定了`decrypt_key`时有效。 | "AES-GCM"、"AES-CBC" | - |
| device | str | `--device=<DEVICE>` | 设置转换模型时的目标设备。使用场景是在Ascend设备上，如果你需要转换生成的模型调用Ascend后端执行推理，则设置该属性，若未设置，默认模型调用CPU后端推理。 | "Ascend" | - |
| encrypt_key | str | `--encryptKey=<ENCRYPTKEY>` | 设置用于加密文件的密钥，以十六进制字符表示。仅支持当`decrypt_mode`是"AES-GCM"，密钥长度为16。 | - | - |
| enable_encryption | bool | `--encryption=<ENCRYPTION>` | 导出模型时是否加密，导出加密可保护模型完整性，但会增加运行时初始化时间。 | True、False | - |
| infer | bool | `--infer=<INFER>` | 是否在转换完成时进行预推理。 | True、False | - |
| input_data_type | DataType | `--inputDataType=<INPUTDATATYPE>` | 设置量化模型输入Tensor的data type。仅当模型输入Tensor的量化参数（`scale`和`zero point`）都具备时有效。默认与原始模型输入Tensor的data type保持一致。 | DataType.FLOAT32、DataType.INT8、DataType.UINT8、DataType.UNKNOWN | - |
| input_format | Format | `--inputDataFormat=<INPUTDATAFORMAT>` | 设置导出模型的输入format，只对四维输入有效。 | Format.NCHW、Format.NHWC | - |
| input_shape | dict{string:list\[int]} | `--inputShape=<INPUTSHAPE>` | 设置模型输入的维度，输入维度的顺序和原始模型保持一致。如：{"inTensor1": \[1, 32, 32, 32], "inTensor2": \[1, 1, 32, 32]} | - | - |
| optimize | str | `--optimize=<OPTIMIZE>` | 设定转换模型的过程所完成的优化。 | "none"、"general"、"gpu_oriented"、"ascend_oriented" | - |
| output_data_type | DataType | `--outputDataType=<OUTPUTDATATYPE>` | 设置量化模型输出Tensor的data type。仅当模型输出Tensor的量化参数（`scale`和`zero point`）都具备时有效。默认与原始模型输出Tensor的data type保持一致。 | DataType.FLOAT32、DataType.INT8、DataType.UINT8、DataType.UNKNOWN | - |
| save_type | ModelType | `--saveType=<SAVETYPE>` | 设置导出模型文件的类型。| ModelType.MINDIR | MINDIR模型使用MindSpore Lite云侧推理安装包 |
| weight_fp16 | bool | `--fp16=<FP16>` | 设置在模型序列化时是否需要将float32数据格式的权重存储为float16数据格式。 | True、False | - |

> - 加解密功能仅在[编译](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/build.html) 时设置为 `MSLITE_ENABLE_MODEL_ENCRYPTION=on` 时生效，并且仅支持Linux x86平台。其中密钥为十六进制表示的字符串，如encrypt_key设置为"30313233343536373839414243444546"，对应的十六进制表示为 `(b)0123456789ABCDEF` ，Linux平台用户可以使用 `xxd` 工具对字节表示的密钥进行十六进制表达转换。需要注意的是，加解密算法在1.7版本进行了更新，导致新版的Python接口不支持对1.6及其之前版本的MindSpore Lite加密导出的模型进行转换。
>
> - `input_shape` 在以下场景下，用户可能需要设置该属性：
>
>   - 用法1：待转换模型的输入是动态shape，准备采用固定shape推理，则设置该属性为固定shape。设置之后，在对Converter后的模型进行推理时，默认输入的shape与该属性设置一样，无需再进行resize操作。
>   - 用法2：无论待转换模型的原始输入是否为动态shape，准备采用固定shape推理，并希望模型的性能尽可能优化，则设置该属性为固定shape。设置之后，将对模型结构进一步优化，但转换后的模型可能会失去动态shape的特征（部分跟shape强相关的算子会被融合）。
>
> - `optimize` 该属性是用来设定在离线转换的过程中需要完成哪些特定的优化。
>
>   - 如果该属性设置为"none"，那么在模型的离线转换阶段将不进行相关的图优化操作，相关的图优化操作将会在执行推理阶段完成。该属性的优点在于转换出来的模型由于没有经过特定的优化，可以直接部署到CPU/GPU/Ascend任意硬件后端；而带来的缺点是推理执行时模型的初始化时间增长。
>   - 如果设置成"general"，表示离线转换过程会完成通用优化，包括常量折叠，算子融合等（转换出的模型只支持CPU/GPU后端，不支持Ascend后端）。
>   - 如果设置成"gpu_oriented"，表示转换过程中会完成通用优化和针对GPU后端的额外优化（转换出来的模型只支持GPU后端）。
>   - 如果设置成"ascend_oriented"，表示转换过程中只完成针对Ascend后端的优化（转换出来的模型只支持Ascend后端）。
>

### convert方法

方法使用场景：将第三方模型转换生成MindSpore模型，可多次调用convert方法，转换多个模型。

下面提供详细的参数说明以及与[推理模型离线转换](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool.html)中参数的对应关系。

| convert方法参数 | 参数类型  | 对应模型离线转换的参数  |  是否必选   |  参数说明  | 取值范围 | 参数默认值 |
| -------- | ----- | -------- | ----- | --- | ---- | ---- |
| fmk_type | FmkType | `--fmk=<FMK>`  | 是 | 输入模型框架类型。 | FmkType.TF、FmkType.CAFFE、FmkType.ONNX、FmkType.TFLITE | - |
| model_file | str | `--modelFile=<MODELFILE>` | 是 | 转换时的输入模型文件路径。 | - | - |
| output_file | str | `--outputFile=<OUTPUTFILE>` | 是 | 转换时的输出模型的路径，可自动生成`.mindir`后缀。 | - | - |
| weight_file | str | `--weightFile=<WEIGHTFILE>` | 转换Caffe模型时必选 | 输入模型权重文件路径。 | - | "" |
| config_file | str | `--configFile=<CONFIGFILE>` | 否 | Converter的配置文件路径，可配置训练后量化或离线拆分算子并行或禁用算子融合功能并将插件设置为so路径等功能。 | - | "" |

> `fmk_type`参数有关详细信息，请参见[FmkType](https://mindspore.cn/lite/api/zh-CN/r2.6.0rc1/mindspore_lite/mindspore_lite.FmkType.html)。
>
> `model_file`举例："/home/user/model.prototxt"。不同类型应模型后缀举例：TF: "model.pb" | CAFFE: "model.prototxt" | ONNX: "model.onnx" | TFLITE: "model.tflite"。
>
> Caffe模型一般分为两个文件：`*.prototxt`是模型结构，对应`model_file`参数；`model.caffemodel`是模型权值，对应`weight_file`参数。
>
> `config_file`配置文件采用`key = value`的方式定义相关参数。
>

### 使用示例

下面选取了几个常用示例，说明转换命令的使用方法。

- 以Caffe模型LeNet为例。

    ```python
    import mindspore_lite as mslite
    converter = mslite.Converter()
    converter.save_type = mslite.ModelType.MINDIR
    converter.optimize = "none"
    converter.convert(fmk_type=mslite.FmkType.CAFFE, model_file="lenet.prototxt",output_file="lenet", weight_file="lenet.caffemodel")
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
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = "general"
        converter.convert(fmk_type=mslite.FmkType.MINDIR, model_file="model.mindir",output_file="model")
        ```

    - TensorFlow Lite模型`model.tflite`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = "none"
        converter.convert(fmk_type=mslite.FmkType.TFLITE, model_file="model.tflite",output_file="model")
        ```

    - TensorFlow模型`model.pb`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = "none"
        converter.convert(fmk_type=mslite.FmkType.TF, model_file="model.pb", output_file="model")
        ```

    - ONNX模型`model.onnx`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = "none"
        converter.convert(fmk_type=mslite.FmkType.ONNX, model_file="model.onnx", output_file="model")
        ```

    以上几种情况下，均显示如下转换成功提示，且同时获得`model.mindir`目标文件。

    ```text
    CONVERT RESULT SUCCESS:0
    ```

### 高级用法

#### 在线转换

get_config_info方法和set_config_info方法用于在线转换，具体请参考[set_config_info](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/mindspore_lite/mindspore_lite.Converter.html#mindspore_lite.Converter.set_config_info)。
