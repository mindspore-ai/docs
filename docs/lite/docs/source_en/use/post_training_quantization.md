# Post Training Quantization

`Windows` `Linux` `Model Converting` `Model Optimization` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/post_training_quantization.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

Converting a trained `float32` model into an `int8` model through quantization after training can reduce the model size and improve the inference performance. In MindSpore Lite, this function is integrated into the model conversion tool `conveter_lite`. You can add command line parameters to convert a model into a quantization model.

MindSpore Lite quantization after training is classified into two types:

1. Weight quantization: quantizes a weight of a model and compresses only the model size. `float32` inference is still performed during inference.
2. Full quantization: quantizes the weight and activation value of a model. The `int` operation is performed during inference to improve the model inference speed and reduce power consumption.

## Configuration Parameter

Post training quantization can be enabled by configuring `configFile` through [Conversion Tool](https://www.mindspore.cn/lite/docs/en/master/use/converter_tool.html). The configuration file adopts the style of `INI`, For quantization, configurable parameters include `common quantization parameter [common_quant_param]`, `mixed bit weight quantization parameter [mixed_bit_weight_quant_param]`,`full quantization parameter [full_quant_param]`, and `data preprocess parameter [data_preprocess_param]`.

### Common Quantization Parameter

common quantization parameters are the basic settings for post training quantization, mainly including `quant_type`, `bit_num`, `min_quant_weight_size`, and `min_quant_weight_channel`. The detailed description of the parameters is as follows:

| Parameter                  | Attribute | Function Description                                         | Parameter Type | Default Value | Value Range                                 |
| -------------------------- | --------- | ------------------------------------------------------------ | -------------- | ------------- | ------------------------------------------- |
| `quant_type`               | Mandatory | The quantization type. When set to WEIGHT_QUANT, weight quantization is enabled; when set to FULL_QUANT, full quantization is enabled. | String         | -             | WEIGHT_QUAN, FULL_QUANT                     |
| `bit_num`                  | Optional  | The number of quantized bits. Currently, weight quantization supports 0-16bit quantization. When it is set to 1-16bit, it is fixed-bit quantization. When it is set to 0bit, mixed-bit quantization is enabled. Full quantization supports 1-8bit quantization. | Integer        | 8             | WEIGHT_QUAN:\[0，16]<br/>FULL_QUANT:\[1，8] |
| `min_quant_weight_size`    | Optional  | Set the threshold of the weight size for quantization. If the number of weights is greater than this value, the weight will be quantized. | Integer        | 0             | [0, 65535]                                  |
| `min_quant_weight_channel` | Optional  | Set the threshold of the number of weight channels for quantization. If the number of weight channels is greater than this value, the weight will be quantized. | Integer        | 16            | [0, 65535]                                  |
| `skip_quant_node`          | Optional | Set the name of the operator that does not need to be quantified, and use `,` to split between multiple operators. | String   | -      | -                                     |
| `debug_info_save_path`     | Optional | Set the folder path where the quantized debug information file is saved. | String   | -      | -                                     |

The common quantization parameter configuration is as follows:

```ini
[common_quant_param]
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=WEIGHT_QUANT
# Weight quantization support the number of bits [0,16], Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
# Full quantization support the number of bits [1,8]
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

> `min_quant_weight_size` and `min_quant_weight_channel` are only valid for weight quantization.
>
> `skip_quant_node` is only valid for full quantization.
>
> Recommendation: When the accuracy of full quantization is not satisfied, you can set `debug_info_save_path` to turn on the Debug mode to get the relevant statistical report, and set `skip_quant_node` for operators that are not suitable for quantization to not quantize them.

### Mixed Bit Weight Quantization Parameter

The mixed bit weight quantization parameters include `init_scale`. When enable the mixed bit weight quantization, the optimal number of bits will be automatically searched for different layers. The detailed description of the parameters is as follows:

| Parameter  | Attribute | Function Description                                         | Parameter Type | Default Value | Value Range |
| ---------- | --------- | ------------------------------------------------------------ | -------------- | ------------- | ----------- |
| init_scale | Optional  | Initialize the scale. The larger the value, the greater the compression rate, but it will also cause varying degrees of accuracy loss. | Float          | 0.02          | (0 , 1)     |
| auto_tune  | Optional  | Automatically search for the init_scale parameter. After setting, it will automatically search for a set of `init_scale` values whose cosine similarity of the model output Tensor is around 0.995. | Boolean        | False         | True，False |

The mixed bit quantization parameter configuration is as follows:

```ini
[mixed_bit_weight_quant_param]
init_scale=0.02
auto_tune=false
```

### Full Quantization Parameters

The full quantization parameters mainly include `activation_quant_method` and `bias_correction`. The detailed description of the parameters is as follows:

| Parameter               | Attribute | Function Description                                | Parameter Type | Default Value | Value Range                                                  |
| ----------------------- | --------- | --------------------------------------------------- | -------------- | ------------- | ------------------------------------------------------------ |
| activation_quant_method | Optional  | Activation quantization algorithm                   | String         | MAX_MIN       | KL, MAX_MIN, or RemovalOutlier.<br/>KL: quantizes and calibrates the data range based on [KL divergence](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf).<br/>MAX_MIN: data quantization parameter computed based on the maximum and minimum values.<br/>RemovalOutlier: removes the maximum and minimum values of data based on a certain proportion and then calculates the quantization parameters.<br/>If the calibration dataset is consistent with the input data during actual inference, MAX_MIN is recommended. If the noise of the calibration dataset is large, KL or RemovalOutlier is recommended. |
| bias_correction         | Optional  | Indicate whether to correct the quantization error. | Boolean        | True          | True or False. After this parameter is enabled, the accuracy of the converted model can be improved. You are advised to set this parameter to true. |

The full quantization parameter configuration is as follows:

```ini
[full_quant_param]
# Activation quantized method supports MAX_MIN or KL or REMOVAL_OUTLIER
activation_quant_method=MAX_MIN
# Whether to correct the quantization error. Recommended to set to true.
bias_correction=true
```

### Data Preprocessing

To calculate the full quantization activation quantized parameter, the user needs to provide a calibration dataset. For the image calibration dataset, data preprocessing functions such as channel conversion, normalization, resize, and center crop will be provided.

| Parameter          | Attribute | Function Description                                         | Parameter Type | Default Value | Value Range                                                  |
| ------------------ | --------- | ------------------------------------------------------------ | -------------- | ------------- | ------------------------------------------------------------ |
| calibrate_path     | Mandatory | The directory where the calibration dataset is stored; if the model has multiple inputs, please fill in the directory where the corresponding data is located one by one, and separate the directory paths with `,` | String         | -             | input_name_1:/mnt/image/input_1_dir,input_name_2:input_2_dir |
| calibrate_size     | Mandatory | Calibration data size                                        | Integer        | -             | [1, 65535]                                                   |
| input_type         | Mandatory | Correction data file format type                             | String         | -             | IMAGE、BIN <br>IMAGE：image file data <br>BIN：binary `.bin` file data |
| image_to_format    | Optional  | Image format conversion                                      | String         | -             | RGB、GRAY、BGR                                               |
| normalize_mean     | Optional  | Normalized mean<br/>dst = (src - mean) / std                 | Vector         | -             | Channel 3: [mean_1, mean_2, mean_3] <br/>Channel 1: [mean_1] |
| normalize_std      | Optional  | Normalized standard deviation<br/>dst = (src - mean) / std   | Vector         | -             | Channel 3: [std_1, std_2, std_3] <br/>Channel 1: [std_1]     |
| resize_width       | Optional  | Resize width                                                 | Integer        | -             | [1, 65535]                                                   |
| resize_height      | Optional  | Resize height                                                | Integer        | -             | [1, 65535]                                                   |
| resize_method      | Optional  | Resize algorithm                                             | String         | -             | LINEAR, NEAREST, CUBIC<br/>LINEAR：Bilinear interpolation<br/>NEARST：Nearest neighbor interpolation<br/>CUBIC：Bicubic interpolation |
| center_crop_width  | Optional  | Center crop width                                            | Integer        | -             | [1, 65535]                                                   |
| center_crop_height | Optional  | Center crop height                                           | Integer        | -             | [1, 65535]                                                   |

The data preprocessing parameter configuration is as follows:

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

## Weight Quantization

Weight quantization supports mixed bit quantization, as well as fixed bit quantization between 1 and 16. The lower the number of bits, the greater the model compression rate, but the accuracy loss is usually larger. The following describes the use and effects of weighting.

### Mixed Bit Weight Quantization

Currently, weight quantization supports mixed bit quantization. According to the distribution of model parameters and the initial value of `init_scale` set by the user, the number of bits that is most suitable for the current layer will be automatically searched out. When the `bit_num` of the configuration parameter is set to 0, mixed bit quantization will be enabled.

The general form of the mixed bit weight requantization command is:

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --quantType=WeightQuant --configFile=/mindspore/lite/tools/converter/quantizer/config/mixed_bit_weight_quant.cfg
```

The mixed bit weight quantification configuration file is as follows:

```ini
[common_quant_param]
quant_type=WEIGHT_QUANT
# Weight quantization support the number of bits [0,16], Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
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

Users can adjust the weighted parameters according to the model and their own needs.

> The init_scale default value is 0.02, and the compression rate is equivalent to the compression effect of 6-7 fixed bits quantization.
>
> Mixed bits need to search for the best bit, and the waiting time may be longer. If you need to view the log, you can set export GLOG_v=1 before the execution to print the relevant Info level log.

### Fixed Bit Weight Quantization

Fixed-bit weighting supports fixed-bit quantization between 1 and 16, and users can adjust the weighting parameters according to the requirement.

The general form of the fixed bit weight quantization conversion command is:

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/fixed_bit_weight_quant.cfg
```

The fixed bit weight quantization configuration file is as follows:

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

### Partial Model Accuracy Result

| Model | Test Dataset | FP32 Model Accuracy | Weight Quantization Accuracy (8 bits) |
| --------            | -------              | -----            | -----     |
| [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) |  77.60%   |   77.53%  |
| [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)      | [ImageNet](http://image-net.org/) |  70.96%  |  70.56% |
| [Mobilenet_V2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz)      | [ImageNet](http://image-net.org/) | 71.56%  |  71.53%  |

> All the preceding results are obtained in the x86 environment.

## Full Quantization

In scenarios where the model running speed needs to be improved and the model running power consumption needs to be reduced, the full quantization after training can be used.

To calculate a quantization parameter of an activation value, you need to provide a calibration dataset. It is recommended that the calibration dataset be obtained from the actual inference scenario and can represent the actual input of a model. The number of data records is about 100.

For image data, currently supports channel pack, normalization, resize, center crop processing. The user can set the corresponding [parameter](https://www.mindspore.cn/lite/docs/en/master/use/post_training_quantization.html#data-preprocessing) according to the preprocessing operation requirements.

The general form of the full quantization conversion command is:

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/full_quant.cfg
```

The full quantization profile is as follows:

```ini
[common_quant_param]
quant_type=FULL_QUANT
# Full quantization support the number of bits [1,8]
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

> Full quantification needs to perform inference, and the waiting time may be longer. If you need to view the log, you can set export GLOG_v=1 before the execution to print the relevant Info level log.

### Partial Model Accuracy Result

| Model                                                        | Test Dataset                      | quant_method | FP32 Model Accuracy | Full Quantization Accuracy (8 bits) | Description                                                  |
| --------            | -------      | -----          | -----            | -----     | -----  |
| [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) | KL       | 77.60%              | 77.40%                              | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |
| [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | [ImageNet](http://image-net.org/) | KL       | 70.96%              | 70.31%                              | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |
| [Mobilenet_V2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) | [ImageNet](http://image-net.org/) | MAX_MIN  | 71.56%              | 71.16%                              | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |

> All the preceding results are obtained in the x86 environment.

## Quantization Debug

Turn on the quantization Debug function, you can get the data distribution statistics report, which is used to evaluate the quantization error and assist the decision-making model (operator) whether it is suitable for quantization. For full quantification, N data distribution statistics reports will be generated according to the number of correction datasets provided, that is, one report will be generated for each round; for weighting, only one data distribution statistics report will be generated.

When setting the `debug_info_save_path` parameter, the relevant debug report will be generated in the `/home/workspace/mindspore/debug_info_save_path` folder:

```ini
[common_quant_param]
debug_info_save_path=/home/workspace/mindspore/debug_info_save_path
```

The data distribution statistics report will count the original data distribution of each Tensor and the data distribution after dequantization of the quantized Tensor. The relevant fields of the data distribution statistics report are as follows:

| Type             | Name                                                         |
| ---------------- | ------------------------------------------------------------ |
| NodeName         | The node name                                                |
| NodeType         | The node type                                                |
| TensorName       | The tensor name                                              |
| InOutFlag        | The input or output tensor                                   |
| DataTypeFlag     | The data type, use Origin for original model, use Dequant for quantization model |
| TensorTypeFlag   | The data types such as input and output, use Activation, and constants, etc., use Weight. |
| Min              | The minimum value                                            |
| Q1               | The 25% quantile                                             |
| Median           | The median                                                   |
| Q3               | The 75% quantile                                             |
| MAX              | The maximum                                                  |
| Mean             | The mean                                                     |
| Var              | The var                                                      |
| Sparsity         | The sparsity                                                 |
| Clip             | The Clip                                                     |
| CosineSimilarity | Cosine similarity compared with the original data            |

The quantization parameter file `quant_param.csv` contains the quantization parameter information of all quantized Tensors. The quantization parameter related fields are as follows:

| Type           | Name                                 |
| -------------- | ------------------------------------ |
| NodeName       | The node name                        |
| NodeType       | The node type                        |
| TensorName     | The tensor name                      |
| ElementsNum    | The Tensor elements num              |
| Dims           | The tensor dims                      |
| Scale          | The quantization parameter scale     |
| ZeroPoint      | The quantization parameter zeropoint |
| Bits           | The number of quantization bits      |
| CorrectionVar  | Bias correction coefficient-variance |
| CorrectionMean | Bias correction coefficient-mean     |

> Mixed bit quantization is non-standard quantization, the quantization parameter file may not exist.

### Skip Quantization Node

Quantization is to convert the Float32 operator to the Int8 operator. The current quantization strategy is to quantify all the nodes contained in a certain type of operator that can be supported, but there are some nodes that are more sensitive and will cause larger errors after quantization. At the same time, the inference speed of some layers after quantization is much lower than that of Float16. It supports non-quantization of the specified layer, which can effectively improve the accuracy and inference speed.

Below is an example of `conv2d_1` `add_8` `concat_1` without quantifying the three nodes:

```ini
[common_quant_param]
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=FULL_QUANT
# Weight quantization support the number of bits [0,16], Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
# Full quantization support the number of bits [1,8]
bit_num=8
# Set the name of the operator that skips the quantization, and use `,` to split between multiple operators.
skip_quant_node=conv2d_1,add_8,concat_1
```

### Recommendations

1. By filtering `InOutFlag == Output && DataTypeFlag == Dequant`, the output layer of all quantization operators can be filtered out, and the accuracy loss of the operator can be judged by looking at the quantized output `CosineSimilarity`, the closer to 1 the smaller the loss.
2. For merging operators such as Add and Concat, if the distribution of `min` and `max` between different input Tensors is quite different, which is likely to cause large errors, you can set `skip_quant_node` to not quantize them.
3. For operators with a higher cutoff rate `Clip`, you can set `skip_quant_node` to not quantize it.