# Optimizing the Model (Quantization After Training)

`Windows` `Linux` `Model Converting` `Model Optimization` `Intermediate` `Expert`

<!-- TOC -->

- [Optimizing the Model (Quantization After Training)](#optimizing-the-model-quantization-after-training)
    - [Overview](#overview)
    - [Configuration Parameter](#configuration-parameter)
        - [Common Quantization Parameter](#common-quantization-parameter)
        - [Mixed Bit Weight Quantization Parameter](#mixed-bit-weight-quantization-parameter)
        - [Full Quantization Parameters](#full-quantization-parameters)
        - [Data Preprocessing](#data-preprocessing)
    - [Weight Quantization](#weight-quantization)
        - [Mixed Bit Weight Quantization](#mixed-bit-weight-quantization)
        - [Fixed Bit Weight Quantization](#fixed-bit-weight-quantization)
        - [Partial Model Accuracy Result](#partial-model-accuracy-result)
    - [Full Quantization](#full-quantization)
        - [Partial Model Accuracy Result](#partial-model-accuracy-result-1)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/lite/docs/source_en/use/post_training_quantization.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Overview

Converting a trained `float32` model into an `int8` model through quantization after training can reduce the model size and improve the inference performance. In MindSpore Lite, this function is integrated into the model conversion tool `conveter_lite`. You can add command line parameters to convert a model into a quantization model.

MindSpore Lite quantization after training is classified into two types:

1. Weight quantization: quantizes a weight of a model and compresses only the model size. `float32` inference is still performed during inference.
2. Full quantization: quantizes the weight and activation value of a model. The `int` operation is performed during inference to improve the model inference speed and reduce power consumption.

## Configuration Parameter

Post training quantization can be enabled by configuring `configFile` through [Conversion Tool](https://www.mindspore.cn/lite/docs/en/r1.5/use/converter_tool.html). The configuration file adopts the style of `INI`, For quantization, configurable parameters include `common quantization parameter [common_quant_param]`, `mixed bit weight quantization parameter [mixed_bit_weight_quant_param]`,`full quantization parameter [full_quant_param]`, and `data preprocess parameter [data_preprocess_param]`.

### Common Quantization Parameter

common quantization parameters are the basic settings for post training quantization, mainly including `quant_type`, `bit_num`, `min_quant_weight_size`, and `min_quant_weight_channel`. The detailed description of the parameters is as follows:

| Parameter                  | Attribute | Function Description                                         | Parameter Type | Default Value | Value Range                                 |
| -------------------------- | --------- | ------------------------------------------------------------ | -------------- | ------------- | ------------------------------------------- |
| `quant_type`               | Mandatory | The quantization type. When set to WEIGHT_QUANT, weight quantization is enabled; when set to FULL_QUANT, full quantization is enabled. | String         | -             | WEIGHT_QUAN, FULL_QUANT                     |
| `bit_num`                  | Optional  | The number of quantized bits. Currently, weight quantization supports 0-16bit quantization. When it is set to 1-16bit, it is fixed-bit quantization. When it is set to 0bit, mixed-bit quantization is enabled. Full quantization supports 1-8bit quantization. | Integer        | 8             | WEIGHT_QUAN:\[0，16]<br/>FULL_QUANT:\[1，8] |
| `min_quant_weight_size`    | Optional  | Set the threshold of the weight size for quantization. If the number of weights is greater than this value, the weight will be quantized. | Integer        | 0             | [0, 65535]                                  |
| `min_quant_weight_channel` | Optional  | Set the threshold of the number of weight channels for quantization. If the number of weight channels is greater than this value, the weight will be quantized. | Integer        | 16            | [0, 65535]                                  |

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
```

### Mixed Bit Weight Quantization Parameter

The mixed bit weight quantization parameters include `init_scale`. When enable the mixed bit weight quantization, the optimal number of bits will be automatically searched for different layers. The detailed description of the parameters is as follows:

| Parameter  | Attribute | Function Description                                         | Parameter Type | Default Value | Value Range |
| ---------- | --------- | ------------------------------------------------------------ | -------------- | ------------- | ----------- |
| init_scale | Optional  | Initialize the scale. The larger the value, the greater the compression rate, but it will also cause varying degrees of accuracy loss. | float          | 0.02          | (0 , 1)     |

The mixed bit quantization parameter configuration is as follows:

```ini
[mixed_bit_weight_quant_param]
init_scale=0.02
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
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=WEIGHT_QUANT
# Weight quantization support the number of bits [0,16], Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
# Full quantization support the number of bits [1,8]
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
> For the sparse structure model, it is recommended to set init_scale to 0.00003.

### Fixed Bit Weight Quantization

Fixed-bit weighting supports fixed-bit quantization between 1 and 16, and users can adjust the weighting parameters according to the requirement.

The general form of the fixed bit weight quantization conversion command is:

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/fixed_bit_weight_quant.cfg
```

The fixed bit weight quantization configuration file is as follows:

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

For image data, currently supports channel pack, normalization, resize, center crop processing. The user can set the corresponding [parameter](#data-preprocessing) according to the preprocessing operation requirements.

The general form of the full quantization conversion command is:

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/full_quant.cfg
```

The full quantization profile is as follows:

```ini
[common_quant_param]
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=FULL_QUANT
# Weight quantization support the number of bits [0,16], Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
# Full quantization support the number of bits [1,8]
bit_num=8
# Layers with size of weights exceeds threshold `min_quant_weight_size` will be quantized.
min_quant_weight_size=0
# Layers with channel size of weights exceeds threshold `min_quant_weight_channel` will be quantized.
min_quant_weight_channel=16

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

### Partial Model Accuracy Result

| Model                                                        | Test Dataset                      | method_x | FP32 Model Accuracy | Full Quantization Accuracy (8 bits) | Description                                                  |
| --------            | -------      | -----          | -----            | -----     | -----  |
| [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) | KL       | 77.60%              | 77.40%                              | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |
| [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | [ImageNet](http://image-net.org/) | KL       | 70.96%              | 70.31%                              | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |
| [Mobilenet_V2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) | [ImageNet](http://image-net.org/) | MAX_MIN  | 71.56%              | 71.16%                              | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |

> All the preceding results are obtained in the x86 environment.
