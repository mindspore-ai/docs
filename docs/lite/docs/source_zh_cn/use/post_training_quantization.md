# 训练后量化

`Windows` `Linux` `模型转换` `模型调优` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/use/post_training_quantization.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

对于已经训练好的`float32`模型，通过训练后量化将其转为`int8`，不仅能减小模型大小，而且能显著提高推理性能。在MindSpore Lite中，这部分功能集成在模型转换工具`conveter_lite`内，通过配置`量化配置文件`的方式，便能够转换得到量化后模型。

MindSpore Lite训练后量化分为两类：

1. 权重量化：对模型的权值进行量化，仅压缩模型大小，推理时仍然执行`float32`推理；
2. 全量化：对模型的权值、激活值等统一进行量化，推理时执行`int`运算，能提升模型推理速度、降低功耗。

## 配置参数

训练后量化可通过[转换工具](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html)配置`configFile`的方式启用训练后量化。配置文件采用`INI`的风格，针对量化场景，目前可配置的参数包括`通用量化参数[common_quant_param]`、`混合比特权重量化参数[mixed_bit_weight_quant_param]`、`全量化参数[full_quant_param]`和`数据预处理参数[data_preprocess_param]`。

### 通用量化参数

通用量化参数是训练后量化的基本设置，主要包括`quant_type`、`bit_num`、`min_quant_weight_size`和`min_quant_weight_channel`。参数的详细介绍如下所示：

| 参数                       | 属性 | 功能描述                                                     | 参数类型 | 默认值 | 取值范围                              |
| -------------------------- | ---- | ------------------------------------------------------------ | -------- | ------ | ------------------------------------- |
| `quant_type`               | 必选 | 设置量化类型，设置为WEIGHT_QUANT时，启用权重量化；设置为FULL_QUANT时，启用全量化；设置为DYNAMIC_QUANT时，启用动态量化。 | String   | -      | WEIGHT_QUANT、FULL_QUANT、DYNAMIC_QUANT |
| `bit_num`                  | 可选 | 设置量化的比特数，目前权重量化支持0-16bit量化，设置为1-16bit时为固定比特量化，设置为0bit时，启用混合比特量化。全量化、动态量化支持8bit量化。 | Integer  | 8      | 权重量化：\[0，16]<br/>全量化：8<br/>动态量化：8 |
| `min_quant_weight_size`    | 可选 | 设置参与量化的权重尺寸阈值，若权重数大于该值，则对此权重进行量化。 | Integer  | 0      | [0, 65535]                            |
| `min_quant_weight_channel` | 可选 | 设置参与量化的权重通道数阈值，若权重通道数大于该值，则对此权重进行量化。 | Integer  | 16     | [0, 65535]                            |
| `skip_quant_node`          | 可选 | 设置无需量化的算子名称，多个算子之间用`,`分割。              | String   | -      | -                                     |
| `debug_info_save_path`     | 可选 | 设置量化Debug信息文件保存的文件夹路径。                      | String   | -      | -                                     |

> 目前`min_quant_weight_size`、`min_quant_weight_channel`仅对权重量化有效。
>
> 建议：全量化在精度不满足的情况下，可设置`debug_info_save_path`开启Debug模式得到相关统计报告，针对不适合量化的算子设置`skip_quant_node`对其不进行量化。

通用量化参数配置如下所示：

```ini
[common_quant_param]
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=WEIGHT_QUANT
# Weight quantization support the number of bits [0,16], Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
# Full quantization support 8bit
bit_num=8
# Layers with size of weights exceeds threshold `min_quant_weight_size` will be quantized.
min_quant_weight_size=0
# Layers with channel size of weights exceeds threshold `min_quant_weight_channel` will be quantized.
min_quant_weight_channel=16
# Set the name of the operator that skips the quantization, and use `,` to split between multiple operators.
skip_quant_node=node_name1,node_name2,node_name3
# Set the folder path where the quantization debug information file is saved.
debug_info_save_path=/home/workspace/mindspore/debug_info_save_path
```

### 混合比特权重量化参数

混合比特权重量化参数包括`init_scale`，启用混合比特权重量化后，将会针对不同层自动搜索最优的比特数。参数的详细介绍如下所示：

| 参数       | 属性 | 功能描述                                                     | 参数类型 | 默认值 | 取值范围    |
| ---------- | ---- | ------------------------------------------------------------ | -------- | ------ | ----------- |
| init_scale | 可选 | 初始化scale，数值越大可以带来更大的压缩率，但是也会造成不同程度的精度损失 | float    | 0.02   | (0 , 1)     |
| auto_tune  | 可选 | 自动搜索init_scale参数，设置后将自动会搜索一组模型输出Tensor在余弦相似度在0.995左右的`init_scale`值 | Boolean  | False  | True，False |

混合比特量化参数配置如下所示：

```ini
[mixed_bit_weight_quant_param]
init_scale=0.02
auto_tune=false
```

### 全量化参数

全量化参数主要包括`activation_quant_method`，`bias_correction` 和`target_device`。参数的详细介绍如下所示：

| 参数                    | 属性 | 功能描述               | 参数类型 | 默认值  | 取值范围                                                     |
| ----------------------- | ---- | ---------------------- | -------- | ------- | ------------------------------------------------------------ |
| activation_quant_method | 可选 | 激活值量化算法         | String   | MAX_MIN | KL，MAX_MIN，RemovalOutlier。 <br>KL：基于[KL散度](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)对数据范围作量化校准。 <br>MAX_MIN：基于最大值、最小值计算数据的量化参数。 <br>RemovalOutlier：按照一定比例剔除数据的极大极小值，再计算量化参数。 <br>在校准数据集与实际推理时的输入数据相吻合的情况下，推荐使用MAX_MIN；而在校准数据集噪声比较大的情况下，推荐使用KL或者REMOVAL_OUTLIER |
| bias_correction         | 可选 | 是否对量化误差进行校正 | Boolean  | True    | True，False。使能后，将能提升量化模型的精度。                |
| target_device         | 可选 | 全量化支持多硬件后端。设置特定硬件后，量化模型会调用专有硬件量化算子库进行推理；如果未设置，转换模型调用通用量化算子库。 | String  | -    | NVGPU: 转换后的量化模型可以在NVIDIA GPU上执行量化推理。|

通用全量化参数配置如下所示：

```ini
[full_quant_param]
# Activation quantized method supports MAX_MIN or KL or REMOVAL_OUTLIER
activation_quant_method=MAX_MIN
# Whether to correct the quantization error. Recommended to set to true.
bias_correction=true
```

NVIDIA GPU全量化参数配置如下：

```ini
[full_quant_param]
# Activation quantized method supports MAX_MIN or KL or REMOVAL_OUTLIER
activation_quant_method=MAX_MIN
# Supports specific hardware backends
target_device=NVGPU
```

### 数据预处理

全量化需要提供100-500张的校准数据集进行预推理，用于计算全量化激活值的量化参数。如果存在多个输入Tensor，每个输入Tensor的校准数据集需要各自保存一个文件夹。

针对BIN格式的校准数据集，`.bin`文件存储的是输入的数据Buffer，同时输入数据的Format需要和推理时输入数据的Format保持一致。针对四维数据，默认是`NHWC`。如果配置了转换工具的命令参数`inputDataFormat`，输入的Buffer的Format需要保持一致。

针对图片格式的校准数据集，后量化提供通道转换、归一化、缩放和裁剪等数据预处理功能。

| 参数               | 属性 | 功能描述                                                     | 参数类型 | 默认值 | 取值范围                                                     |
| ------------------ | ---- | ------------------------------------------------------------ | -------- | ------ | ------------------------------------------------------------ |
| calibrate_path     | 必选 | 存放校准数据集的目录；如果模型有多个输入，请依次填写对应的数据所在目录，目录路径间请用`,`隔开 | String   | -      | input_name_1:/mnt/image/input_1_dir,input_name_2:input_2_dir |
| calibrate_size     | 必选 | 矫正集数量                                                   | Integer  | -      | [1, 65535]                                                   |
| input_type         | 必选 | 矫正数据文件格式类型                                         | String   | -      | IMAGE、BIN <br>IMAGE：图片文件数据 <br>BIN：满足推理要求的输入二进制`.bin`文件数据 |
| image_to_format    | 可选 | 图像格式转换                                                 | String   | -      | RGB、GRAY、BGR                                               |
| normalize_mean     | 可选 | 图像归一化的均值<br/>dst = (src - mean) / std                | Vector   | -      | 3通道：[mean_1, mean_2, mean_3] <br/>1通道：[mean_1]         |
| normalize_std      | 可选 | 图像归一化的标准差<br/>dst = (src - mean) / std              | Vector   | -      | 3通道：[std_1, std_2, std_3] <br/>1通道：[std_1]             |
| resize_width       | 可选 | 图像缩放宽度                                                 | Integer  | -      | [1, 65535]                                                   |
| resize_height      | 可选 | 图像缩放高度                                                 | Integer  | -      | [1, 65535]                                                   |
| resize_method      | 可选 | 图像缩放算法                                                 | String   | -      | LINEAR、NEAREST、CUBIC<br/>LINEAR：线性插值<br/>NEARST：最邻近插值<br/>CUBIC：三次样条插值 |
| center_crop_width  | 可选 | 中心裁剪宽度                                                 | Integer  | -      | [1, 65535]                                                   |
| center_crop_height | 可选 | 中心裁剪高度                                                 | Integer  | -      | [1, 65535]                                                   |

数据预处理参数配置如下所示：

```ini
[data_preprocess_param]
# Calibration dataset path, the format is input_name_1:input_1_dir,input_name_2:input_2_dir
# Full quantification must provide correction dataset
calibrate_path=input_name_1:/mnt/image/input_1_dir,input_name_2:input_2_dir
# Calibration data size
calibrate_size=100
# Input type supports IMAGE or BIN
# When set to IMAGE, the image data will be read
# When set to BIN, the `.bin` binary file will be read
input_type=IMAGE
# The output format of the preprocessed image
# Supports RGB or GRAY or BGR
image_to_format=RGB
# Image normalization
# dst = (src - mean) / std
normalize_mean=[127.5, 127.5, 127.5]
normalize_std=[127.5, 127.5, 127.5]
# Image resize
resize_width=224
resize_height=224
# Resize method supports LINEAR or NEAREST or CUBIC
resize_method=LINEAR
# Image center crop
center_crop_width=224
center_crop_height=224
```

## 权重量化

权重量化支持混合比特量化，同时也支持1~16之间的固定比特量化，比特数越低，模型压缩率越大，但是精度损失通常也比较大。下面对权重量化的使用方式和效果进行阐述。

### 混合比特量化

目前权重量化支持混合比特量化，会根据模型参数的分布情况，根据用户设置的`init_scale`作为初始值，自动搜索出最适合当前层的比特数。配置参数的`bit_num`设置为0时，将启用混合比特量化。

混合比特权重量化转换命令的一般形式为：

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/mixed_bit_weight_quant.cfg
```

混合比特权重量化配置文件如下所示：

```ini
[common_quant_param]
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=WEIGHT_QUANT
# Weight quantization support the number of bits [0,16], Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
# Full quantization support 8bit
bit_num=0
# Layers with size of weights exceeds threshold `min_quant_weight_size` will be quantized.
min_quant_weight_size=5000
# Layers with channel size of weights exceeds threshold `min_quant_weight_channel` will be quantized.
min_quant_weight_channel=5

[mixed_bit_weight_quant_param]
# Initialization scale in (0,1).
# A larger value can get a larger compression ratio, but it may also cause a larger error.
init_scale=0.02
```

用户可根据模型及自身需要对权重量化的参数作出调整。
> init_scale默认的初始值为0.02，搜索的压缩率相当与6-7固定比特的压缩效果。
>
> 混合比特需要搜索最佳比特位，等待时间可能较长，如果需要查看日志，可以在执行前设置export GLOG_v=1，用于打印相关Info级别日志。

### 固定比特量化

固定比特的权重量化支持1~16之间的固定比特量化，用户可根据模型及自身需要对权重量化的参数作出调整。

固定比特权重量化转换命令的一般形式为：

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/fixed_bit_weight_quant.cfg
```

固定比特权重量化配置文件如下所示：

```ini
[common_quant_param]
quant_type=WEIGHT_QUANT
# Weight quantization support the number of bits [0,16], Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
bit_num=8
# Layers with size of weights exceeds threshold `min_quant_weight_size` will be quantized.
min_quant_weight_size=0
# Layers with channel size of weights exceeds threshold `min_quant_weight_channel` will be quantized.
min_quant_weight_channel=16
```

### 部分模型精度结果

|  模型                |  测试数据集        |  FP32模型精度    |  权重量化精度（8bit） |
| --------            | -------              | -----            | -----     |
| [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) |  77.60%   |   77.53%  |
| [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)      | [ImageNet](http://image-net.org/) |  70.96%  |  70.56% |
| [Mobilenet_V2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz)      | [ImageNet](http://image-net.org/) | 71.56%  |  71.53%  |

> 以上所有结果均在x86环境上测得。

## 全量化

针对CV模型需要提升模型运行速度、降低模型运行功耗的场景，可以使用训练后全量化功能。下面对全量化的使用方式和效果进行阐述。

全量化计算激活值的量化参数，用户需要提供校准数据集。校准数据集最好来自真实推理场景，能表征模型的实际输入情况，数量在100 - 500个左右。

针对图片数据，目前支持通道调整、归一化、缩放、裁剪等预处理的功能。用户可以根据推理时所需的预处理操作，设置相应的[参数](https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html#数据预处理)。

注意：

- 模型校准数据必须与训练数据同分布，并且校准数据与模型输入的format（例如：NCHW、NHWC）需要保持一致。

全量化转换命令的一般形式为：

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/full_quant.cfg
```

全量化配置文件如下所示：

```ini
[common_quant_param]
quant_type=FULL_QUANT
# Full quantization support 8bit
bit_num=8

[data_preprocess_param]
# Calibration dataset path, the format is input_name_1:input_1_dir,input_name_2:input_2_dir
# Full quantification must provide correction dataset
calibrate_path=input_name_1:/mnt/image/input_1_dir,input_name_2:input_2_dir
# Calibration data size
calibrate_size=100
# Input type supports IMAGE or BIN
# When set to IMAGE, the image data will be read
# When set to BIN, the `.bin` binary file will be read
input_type=IMAGE
# The output format of the preprocessed image
# Supports RGB or GRAY or BGR
image_to_format=RGB
# Image normalization
# dst = (src - mean) / std
normalize_mean=[127.5, 127.5, 127.5]
normalize_std=[127.5, 127.5, 127.5]
# Image resize
resize_width=224
resize_height=224
# Resize method supports LINEAR or NEAREST or CUBIC
resize_method=LINEAR
# Image center crop
center_crop_width=224
center_crop_height=224

[full_quant_param]
# Activation quantized method supports MAX_MIN or KL or REMOVAL_OUTLIER
activation_quant_method=MAX_MIN
# Whether to correct the quantization error. Recommended to set to true.
bias_correction=true
```

> 全量化需要执行推理，等待时间可能较长，如果需要查看日志，可以在执行前设置export GLOG_v=1，用于打印相关Info级别日志。

### 部分模型精度结果

|  模型                |  测试数据集   | 量化方法    |  FP32模型精度    |  全量化精度（8bit） | 说明 |
| --------            | -------      | -----          | -----            | -----     | -----  |
| [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) | KL |    77.60%   |   77.40%   | 校准数据集随机选择ImageNet Validation数据集中的100张 |
| [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)      | [ImageNet](http://image-net.org/) | KL |    70.96%    |  70.31%  | 校准数据集随机选择ImageNet Validation数据集中的100张 |
| [Mobilenet_V2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz)      | [ImageNet](http://image-net.org/) | MAX_MIN |    71.56%    |  71.16%  | 校准数据集随机选择ImageNet Validation数据集中的100张 |

> 以上所有结果均在x86环境上测得。

## 动态量化

针对NLP模型需要提升模型运行速度、降低模型运行功耗的场景，可以使用动态量化功能。下面对动态量化的使用方式和效果进行阐述。

动态量化的权重是离线转换阶段量化，而激活是在运行阶段才进行量化。

动态量化转换命令的一般形式为：

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/dynamic_quant.cfg
```

动态量化配置文件如下所示：

```ini
[common_quant_param]
quant_type=DYNAMIC_QUANT
bit_num=8
```

> 为了保证量化精度，目前动态量化不支持设置FP16的运行模式。
>
> 目前动态量化在支持SDOT指令的ARM架构会有进一步的加速效果。

### 部分模型性能结果

- tinybert_encoder

| 模型类型            | 运行模式      | Model Size(M) | RAM(K)     | Latency(ms) | Cos-Similarity | 压缩率      | 内存相比FP32 | 时延相比FP32 |
| ------------------- | ------------- | ------------- | ---------- | ----------- | -------------- | ----------- | ------------ | ------------ |
| FP32                | FP32          | 20            | 29,029     | 9.916       | 1              |             |              |              |
| FP32                | FP16          | 20            | 18,208     | 5.75        | 0.99999        | 1           | -37.28%      | -42.01%      |
| FP16                | FP16          | 12            | 18,105     | 5.924       | 0.99999        | 1.66667     | -37.63%      | -40.26%      |
| Weight Quant(8 Bit) | FP16          | 5.3           | 19,324     | 5.764       | 0.99994        | 3.77358     | -33.43%      | -41.87%      |
| **Dynamic Quant**   | **INT8+FP32** | **5.2**       | **15,709** | **4.517**   | **0.99668**    | **3.84615** | **-45.89%**  | **-54.45%**  |

- tinybert_decoder

| 模型类型            | 运行模式      | Model Size(M) | RAM(K)     | Latency(ms) | Cos-Similarity | 压缩率      | 内存相比FP32 | 时延相比FP32 |
| ------------------- | ------------- | ------------- | ---------- | ----------- | -------------- | ----------- | ------------ | ------------ |
| FP32                | FP32          | 43            | 51,355     | 4.161       | 1              |             |              |              |
| FP32                | FP16          | 43            | 29,462     | 2.184       | 0.99999        | 1           | -42.63%      | -47.51%      |
| FP16                | FP16          | 22            | 29,440     | 2.264       | 0.99999        | 1.95455     | -42.67%      | -45.59%      |
| Weight Quant(8 Bit) | FP16          | 12            | 32,285     | 2.307       | 0.99998        | 3.58333     | -37.13%      | -44.56%      |
| **Dynamic Quant**   | **INT8+FP32** | **12**        | **22,181** | **2.074**   | **0.9993**     | **3.58333** | **-56.81%**  | **-50.16%**  |

## 量化Debug

开启量化Debug功能，能够得到数据分布统计报告，用于评估量化误差，辅助决策模型（算子）是否适合量化。针对全量化，会根据所提供矫正数据集的数量，生成N份数据分布统计报告，即每一轮都会生成一份报告；针对权重量化，只会生成1份数据分布统计报告。

设置`debug_info_save_path`参数后，将会在`/home/workspace/mindspore/debug_info_save_path`文件夹中生成相关Debug报告：

```ini
[common_quant_param]
debug_info_save_path=/home/workspace/mindspore/debug_info_save_path
```

数据分布统计报告会统计每个Tensor原始数据分布以及量化Tensor反量化后的数据分布情况。数据分布统计报告相关字段如下所示：

| Type             | Name                                                     |
| ---------------- | -------------------------------------------------------- |
| NodeName         | 节点名                                                   |
| NodeType         | 节点类型                                                 |
| TensorName       | Tensor名                                                 |
| InOutFlag        | Tensor输出、输出类型                                     |
| DataTypeFlag     | 数据类型，原始数据用Origin，反量化后的数据用Dequant      |
| TensorTypeFlag   | 针对输入输出等数据类用Activation表示，常量等用Weight表示 |
| Min              | 最小值，0%分位点                                         |
| Q1               | 25%分位点                                                |
| Median           | 中位数，50%分位点                                        |
| Q3               | 75%分位点                                                |
| MAX              | 最大值，100%分位点                                       |
| Mean             | 均值                                                     |
| Var              | 方差                                                     |
| Sparsity         | 稀疏度                                                   |
| Clip             | 截断率                                                   |
| CosineSimilarity | 和原始数据对比的余弦相似度                               |

量化参数文件`quant_param.csv`包含所有量化Tensor的量化参数信息，量化参数相关字段如下所示：

| Type           | Name              |
| -------------- | ----------------- |
| NodeName       | 节点名            |
| NodeType       | 节点类型          |
| TensorName     | Tensor名          |
| ElementsNum    | Tensor数据量      |
| Dims           | Tensor维度        |
| Scale          | 量化参数scale     |
| ZeroPoint      | 量化参数ZeroPoint |
| Bits           | 量化比特数        |
| CorrectionVar  | 误差矫正系数-方差 |
| CorrectionMean | 误差矫正系数-均值 |

> 由于混合比特量化是非标准量化，该量化参数文件可能不存在。

### 设置无需量化Node

量化是将Float32算子转换Int8算子，目前的量化策略是针对可支持的某一类算子所包含的Node都会进行量化，但是存在部分Node敏感度较高，量化后会引发较大的误差，同时某些层量化后推理速度远低于Float16的推理速度。支持指定层不量化，可以有效提高精度和推理速度。

下面将`conv2d_1` `add_8` `concat_1`三个Node不进行量化的示例：

```ini
[common_quant_param]
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=FULL_QUANT
# Weight quantization support the number of bits [0,16], Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
# Full quantization support 8bit
bit_num=8
# Set the name of the operator that skips the quantization, and use `,` to split between multiple operators.
skip_quant_node=conv2d_1,add_8,concat_1
```

### 使用建议

1. 通过过滤`InOutFlag == Output && DataTypeFlag == Dequant`，可以筛选出所有量化算子的输出层，通过查看量化输出的`CosineSimilarity`来判断算子的精度损失，越接近1损失越小。
2. 针对Add、Concat等合并类算子，如果不同输入Tensor之间`min`、`max`分布差异较大，容易引发较大误差，可以设置`skip_quant_node`，将其不量化。
3. 针对截断率`Clip`较高的算子，可以设置`skip_quant_node`，将其不量化。
