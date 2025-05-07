# Quantization

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/docs/lite/docs/source_en/advanced/quantization.md)

## Overview

Converting a trained `float32` model into an `int8` model through quantization after training can reduce the model size and improve the inference performance. In MindSpore Lite, this can be achieved by using the model conversion tool `converter_lite`, which configures the `quantization profile` and then converts the quantized model.

## Quantization Algorithm

There are two kinds of quantization algorithms: quantization-aware training and post-training quantization. MindSpore Lite supports post-training quantization.

MindSpore Lite post-training quantization currently supports three specific algorithms with the following specifications:

| Algorithm Types | Quantification of Weights | Quantification of Activation |  Compression Effect of Model Size | Inference Acceleration Effect | Precision Loss Effect |
| -------- | ---------- | ---------- | ---------------- | ------------ | ------------ |
| weight quantification | Y         | N         | Excellent               | Average         | Excellent           |
| full quantization   | Y         | Y         | Excellent               | Good           | Average         |
| dynamic quantification | Y         | Y         | Excellent               | Good         | Good         |

### Weight Quantization

Weight quantization supports mixed bit quantization, as well as fixed bit quantization between 1 and 16 and ON_THE_FLY quantization. The lower the number of bits, the greater the model compression rate, but the loss of accuracy is usually also great. The following describes the use and effect of weighted quantization.

#### Mixed Bit Quantization

Mixed bit quantization automatically searches for the most appropriate number of bits for the current layer based on the distribution of model parameters, using the user-set `init_scale` as the initial value. Mixed bit quantization will be enabled when `bit_num` of the configuration parameter is set to 0.

The general form of the mixed bit quantization conversion command is:

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/mixed_bit_weight_quant.cfg
```

The mixed bit quantization configuration profile is shown below:

```ini
[common_quant_param]
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=WEIGHT_QUANT
# Weight quantization supports the number of bits [0,16]. Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
# Full quantization supports 8bit
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

Users can adjust the parameters of weight quantization according to the model and their needs.
> The default initial value of init_scale is 0.02, which searches for a compression rate comparable to that of 6-7 fixed bits.
>
> Mixed bits need to search for the best bits, the waiting time may be long, if you need to view the log, you can set export GLOG_v=1 before execution for printing the related Info level log.

#### Fixed Bit Quantization

Fixed bit weight quantization supports fixed bit quantization between 1 and 16, and users can adjust the parameters of weight quantization according to the model and their own needs.

The general form of the mixed bit quantization conversion command is:

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/fixed_bit_weight_quant.cfg
```

The fixed bit quantization configuration profile is shown below:

```ini
[common_quant_param]
quant_type=WEIGHT_QUANT
# Weight quantization supports the number of bits [0,16]. Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
bit_num=8
# Layers with size of weights exceeds threshold `min_quant_weight_size` will be quantized.
min_quant_weight_size=0
# Layers with channel size of weights exceeds threshold `min_quant_weight_channel` will be quantized.
min_quant_weight_channel=16
```

#### ON_THE_FLY Quantization

Ascend ON_THE_FLY quantization indicates runtime weight inverse quantization. Only Ascend inference is supported at this stage.

Ascend ON_THE_FLY quantization also requires a new `[ascend_context]` related configuration, the Ascend ON_THE_FLY quantization configuration file is shown below:

```ini
[common_quant_param]
quant_type=WEIGHT_QUANT
# Weight quantization supports the number of bits (0,16]
bit_num=8
# Layers with size of weights exceeds threshold `min_quant_weight_size` will be quantized.
min_quant_weight_size=5000
# Layers with channel size of weights exceeds threshold `min_quant_weight_channel` will be quantized.
min_quant_weight_channel=5

[weight_quant_param]
dequant_strategy=ON_THE_FLY
# If set to true, it will enable PerChannel quantization, or set to false to enable PerLayer quantization.
per_channel=True
# Whether to correct the quantization error. Recommended to set to true.
bias_correction=False

[ascend_context]
# The converted model is suitable for Ascend GE processes
provider=ge
```

### Full Quantization

For the scenarios where the CV model needs to improve the model running speed and reduce the power consumption of the model running, the post-training full quantization can be used. The following describes the use and effect of full quantization.

To fully quantize the quantization parameters for calculating the activation values, the user needs to provide a calibration dataset. The calibration dataset should preferably come from real inference scenarios that characterize the actual inputs to the model, in the order of 100 - 500, **and the calibration dataset needs to be processed into `NHWC` format**.

For image data, it currently supports the functions of channel adjustment, normalization, scaling, cropping and other preprocessing. The user can set the appropriate [Data Preprocessing Parameters](https://www.mindspore.cn/lite/docs/en/r2.6.0/advanced/quantization.html#data-preprocessing-parameters) according to the preprocessing operation required for inference.

User configuration of full quantization requires at least `[common_quant_param]`, `[data_preprocess_param]`, and `[full_quant_param]`.

> The model calibration data must be co-distributed with the training data, and the Format of the calibration data and that of the inputs of the exported floating-point model need to be consistent.

The general form of the full quantization conversion command is:

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/full_quant.cfg
```

#### CPU

The complete configuration file of full CPU quantization is shown below:

```ini
[common_quant_param]
quant_type=FULL_QUANT
# Full quantization supports 8bit
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

> Full quantization requires the execution of inference, the waiting time may be long, if you need to view the log, you can set export GLOG_v=1 before execution for printing the related Info level log.

The full quantization (weighted PerChannel quantization method) parameter `[full_quant_param]` is configured as shown below:

```ini
[full_quant_param]
# Activation quantized method supports MAX_MIN or KL or REMOVAL_OUTLIER
activation_quant_method=MAX_MIN
# Whether to correct the quantization error. Recommended to set to true.
bias_correction=true
# Enable PerChannel quantization.
per_channel=true
```

The full quantization (weighted PerLayer quantization method) parameter `[full_quant_param]` is configured as shown below:

```ini
[full_quant_param]
# Activation quantized method supports MAX_MIN or KL or REMOVAL_OUTLIER
activation_quant_method=MAX_MIN
# Whether to correct the quantization error. Recommended to set to true.
bias_correction=true
# Enable PerLayer quantization.
per_channel=false
```

#### NVIDIA GPU

NVIDIA GPU full quantization parameter configuration, just add a new configuration `target_device=NVGPU` to `[full_quant_param]`:

```ini
[full_quant_param]
# Activation quantized method supports MAX_MIN or KL or REMOVAL_OUTLIER
activation_quant_method=MAX_MIN
# Supports specific hardware backends
target_device=NVGPU
```

#### DSP

DSP full quantization parameter configuration, just add a new configuration `target_device=DSP` to `[full_quant_param]`:

```ini
[full_quant_param]
# Activation quantized method supports MAX_MIN or KL or REMOVAL_OUTLIER
activation_quant_method=MAX_MIN
# Whether to correct the quantization error.
bias_correction=false
# Supports specific hardware backends
target_device=DSP
```

#### Ascend

Ascend quantization needs to configure Ascend-related configuration at [offline conversion](https://www.mindspore.cn/lite/docs/en/r2.6.0/mindir/converter_tool.html#description-of-parameters) first, i.e. `optimize` needs to be set to `ascend_oriented`, and then configure Ascend related environment variables during conversion.

**Ascend Fully Quantized Static Shape Parameter Configuration**

- The general form of the conversion command for Ascend-related environment variables in the Ascend fully quantized static shape scenario is:

    ```bash
    ./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/full_quant.cfg --optimize=ascend_oriented
    ```

- In static shape scenario, Ascend fully quantized parameter just adds a new configuration `target_device=ASCEND` to `[full_quant_param]`.

    ```ini
    [full_quant_param]
    # Activation quantized method supports MAX_MIN or KL or REMOVAL_OUTLIER
    activation_quant_method=MAX_MIN
    # Whether to correct the quantization error.
    bias_correction=true
    # Supports specific hardware backends
    target_device=ASCEND
    ```

**Ascend full quantization supports dynamic Shape parameters**. The conversion command needs to set the same inputShape of the calibration dataset, which can be found in [Conversion Tool Parameter Description](https://www.mindspore.cn/lite/docs/en/r2.6.0/mindir/converter_tool.html#description-of-parameters).

- The general form of the conversion command in the Ascend fully quantized static shape scenario is:

    ```bash
    ./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/full_quant.cfg --optimize=ascend_oriented --inputShape="inTensorName_1:1,32,32,4"
    ```

- Ascend fully quantized parameter dynamic shape scenarios also needs to add new `[ascend_context]` related configurations.

    ```ini
    [full_quant_param]
    # Activation quantized method supports MAX_MIN or KL or REMOVAL_OUTLIER
    activation_quant_method=MAX_MIN
    # Whether to correct the quantization error.
    bias_correction=true
    # Supports specific hardware backends
    target_device=ASCEND

    [ascend_context]
    input_shape=input_1:[-1,32,32,4]
    dynamic_dims=[1~4],[8],[16]

    # where "-1" in input_shape means the dynamic batch is set.
    ```

### Dynamic Quantization

For the scenarios where the NLP model needs to improve the model running speed and reduce the model running power consumption, the dynamic quantization function can be used. The use and effect of dynamic quantization are described below.

The weights for dynamic quantization are quantified in the offline conversion phase, whereas activation is quantified only in the runtime phase.

The general form of the dynamic quantization conversion command is:

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/dynamic_quant.cfg
```

The dynamic quantization profile is shown below:

```ini
[common_quant_param]
quant_type=DYNAMIC_QUANT
bit_num=8

[dynamic_quant_param]
# If set to ALWC, it will enable activation perlayer and weight perchannel quantization. If set to ACWL, it will enable activation perchannel and weight perlayer quantization. Default value is ALWC.
quant_strategy=ACWL
```

> In order to ensure quantization accuracy, dynamic quantization currently does not support setting the operation mode of the FP16.
>
> Currently dynamic quantization will be further accelerated in ARM architectures that support SDOT instructions.

## Configuration Parameter

Post training quantization can be enabled by configuring `configFile` through [Conversion Tool](https://www.mindspore.cn/lite/docs/en/r2.6.0/converter/converter_tool.html). The configuration file adopts the style of [`INI`](https://en.wikipedia.org/wiki/INI_file). For quantization, configurable parameters include:

- `[common_quant_param]: Public quantization parameters`
- `[weight_quant_param]: Fixed bit quantization parameters`
- `[mixed_bit_weight_quant_param]: Mixed bit quantization parameters`
- `[full_quant_param]: Full quantization parameters`
- `[data_preprocess_param]: Data preprocessing parameters`
- `[dynamic_quant_param]: Dynamic quantization parameters`

### Common Quantization Parameter

common quantization parameters are the basic settings for post training quantization. The detailed description of the parameters is as follows:

| Parameter                  | Attribute | Function Description                                         | Parameter Type | Default Value | Value Range                                 |
| -------------------------- | --------- | ------------------------------------------------------------ | -------------- | ------------- | ------------------------------------------- |
| `quant_type`               | Mandatory | The quantization type. When set to WEIGHT_QUANT, weight quantization is enabled; when set to FULL_QUANT, full quantization is enabled; when set to DYNAMIC_QUANT, dynamic quantization is enabled. | String         | -             | WEIGHT_QUANT,<br/> FULL_QUANT,<br/>DYNAMIC_QUANT |
| `bit_num`                  | Optional  | The number of quantized bits. Currently, weight quantization supports 0-16bit quantization. When it is set to 1-16bit, it is fixed-bit quantization. When it is set to 0bit, mixed-bit quantization is enabled. Full quantization and Dynamic quantization supports 8bit quantization. | Integer        | 8             | WEIGHT_QUANT:\[0, 16]<br/>FULL_QUANT: 8<br/>DYNAMIC_QUANT:8 |
| `min_quant_weight_size`    | Optional  | Set the threshold of the weight size for quantization. If the number of weights is greater than this value, the weight will be quantized. | Integer        | 0             | [0, 65535]                                  |
| `min_quant_weight_channel` | Optional  | Set the threshold of the number of weight channels for quantization. If the number of weight channels is greater than this value, the weight will be quantized. | Integer        | 16            | [0, 65535]                                  |
| `skip_quant_node`          | Optional | Set the name of the operator that does not need to be quantified, and use `,` to split between multiple operators. | String   | -      | -                                     |
| `debug_info_save_path`     | Optional | Set the folder path where the quantized debug information file is saved. | String   | -      | -                                     |
| `enable_encode`     | Optional |  The enable switch of compression code for weight quantization. | Boolean   | True     |  True, False                                    |

> `min_quant_weight_size` and `min_quant_weight_channel` are only valid for weight quantization.
>
> Recommendation: When the accuracy of full quantization is not satisfied, you can set `debug_info_save_path` to turn on the Debug mode to get the relevant statistical report, and set `skip_quant_node` for operators that are not suitable for quantization to not quantize them.

The common quantization parameter configuration is as follows:

```ini
[common_quant_param]
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=WEIGHT_QUANT
# Weight quantization supports the number of bits [0,16]. Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
# Full quantization supports 8bit
bit_num=8
# Layers with size of weights exceeds threshold `min_quant_weight_size` will be quantized.
min_quant_weight_size=0
# Layers with channel size of weights exceeds threshold `min_quant_weight_channel` will be quantized.
min_quant_weight_channel=16
# Set the name of the operator that skips the quantization, and use `,` to split between multiple operators.
skip_quant_node=node_name1,node_name2,node_name3
# Set the folder path where the quantization debug information file is saved.
debug_info_save_path=/home/workspace/mindspore/debug_info_save_path
# Enable tensor compression for weight quantization. If parameter bit_num not equal to 8 or 16, it can not be set to false.
enable_encode = true
```

### Fixed Bit Quantization Parameters

The detailed description of the fixed bit quantization parameters is as follows:

| Parameter  | Attribute | Function Description                                | Parameter Type | Default Value | Value Range                                                                                             |
|------------------|----|-----------------------------------------------------|---------|------|---------------------------------------------------------------------------------------------------------|
| init_scale | optional | Initialize the scale. Larger values will result in greater compression, but will also result in varying degrees of precision loss. | float    | 0.02   | (0 , 1)     |
| auto_tune  | optional | The `init_scale` parameter will automatically search for a set of `init_scale` values for which the model output Tensor has a cosine similarity around 0.995 after setting. | Boolean  | False  | True, False |

The mixed bit quantization parameter configuration is shown below:

```ini
[mixed_bit_weight_quant_param]
init_scale=0.02
auto_tune=false
```

### Mixed Bit Quantization Parameter

The details of the fixed bit quantization parameters are shown below:

| Parameter  | Attribute | Function Description                                         | Parameter Type | Default Value | Value Range |
| ---------- | --------- | ------------------------------------------------------------ | -------------- | ------------- | ----------- |
| per_channel     | optional | Quantization by PerChannel or PerLayer | Boolean  | True   | True, False. set to False to enable the PerLayer quantization method. |
| bias_correction | optional | Whether to correct for quantization errors             | Boolean  | True   | True, False. If it is enabled, the accuracy of the quantization model will be improved.      |

The mixed bit quantization parameter configuration is as follows:

```ini
[weight_quant_param]
# If set to true, it will enable PerChannel quantization, or set to false to enable PerLayer quantization.
per_channel=True
# Whether to correct the quantization error. Recommended to set to true.
bias_correction=False
```

### ON_THE_FLY Quantization Parameters

The detailed description of the ON_THE_FLY quantization parameters is as follows:

| Parameter               | Attribute | Function Description                                | Parameter Type | Default Value | Value Range                                                  |
| ----------------------- | --------- | --------------------------------------------------- | -------------- | ------------- | ------------------------------------------------------------ |
| dequant_strategy | optional | Weight quantification model | String   | -      | ON_THE_FLY. If it is enabled, the Ascend online inverse quantization mode is enabled. |

The ON_THE_FLY quantization parameter is configured as shown below:

```ini
[weight_quant_param]
# Enable ON_THE_FLY quantization
dequant_strategy=ON_THE_FLY

[ascend_context]
# The converted model is suitable for Ascend GE processes
provider=ge
```

### Full Quantization Parameters

The detailed description of the full quantization parameters is as follows:

| Parameter               | Attribute | Function Description                                | Parameter Type | Default Value | Value Range                                                  |
| ----------------------- | --------- | --------------------------------------------------- | -------------- | ------------- | ------------------------------------------------------------ |
| activation_quant_method | Optional  | Activation quantization algorithm                   | String         | MAX_MIN       | KL, MAX_MIN, or RemovalOutlier.<br/>KL: quantizes and calibrates the data range based on [KL divergence](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf).<br/>MAX_MIN: data quantization parameter computed based on the maximum and minimum values.<br/>RemovalOutlier: removes the maximum and minimum values of data based on a certain proportion and then calculates the quantization parameters.<br/>If the calibration dataset is consistent with the input data during actual inference, MAX_MIN is recommended. If the noise of the calibration dataset is large, KL or RemovalOutlier is recommended. |
| bias_correction         | Optional  | Indicate whether to correct the quantization error. | Boolean        | True          | True or False. After this parameter is enabled, the accuracy of the converted model can be improved. You are advised to set this parameter to true. |
| per_channel         | Optional  | Select PerChannel or PerLayer quantization type. | Boolean        | True          | True or False. Set to false to enable Perlayer quantization. |
| target_device         | Optional  | Full quantization supports multiple hardware backends. After setting the specific hardware, the converted quantization model can execute the proprietry hardware quantization operator library. If not setting, universal quantization lib will be called. | String        | -          | NVGPU: The quantized model can perform quantitative inference on the NVIDIA GPU. <br/>DSP: The quantized model can perform quantitative inference on the DSP devices.<br/>ASCEND: The quantized model can perform quantitative inference on the ASCEND devices. |

The fully quantization parameter configuration is shown below:

```ini
[full_quant_param]
# Activation quantized method supports MAX_MIN or KL or REMOVAL_OUTLIER
activation_quant_method=MAX_MIN
# Whether to correct the quantization error. Recommended to set to true.
bias_correction=true
# Enable PerChannel quantization.
per_channel=true
# Supports specific hardware backends
target_device=NVGPU
```

### Data Preprocessing Parameters

Full quantization needs to provide 100-500 calibration data sets for pre-inference, which is used to calculate the quantization parameters of full quantization activation values. If there are multiple input Tensors, the calibration dataset for each input Tensor needs to be saved in a separate folder.

For the BIN calibration dataset, the `.bin` file stores the input data buffer, and the format of the input data needs to be consistent with the format of the input data during inference. For 4D data, the default is `NHWC`. If the command parameter `inputDataFormat` of the converter tool is configured, the format of the input Buffer needs to be consistent.

For the image calibration dataset, post training quantization provides data preprocessing functions such as channel conversion, normalization, resize, and center crop.

| Parameter          | Attribute | Function Description                                         | Parameter Type | Default Value | Value Range                                                  |
| ------------------ | --------- | ------------------------------------------------------------ | -------------- | ------------- | ------------------------------------------------------------ |
| calibrate_path     | Mandatory | The directory where the calibration dataset is stored; if the model has multiple inputs, please fill in the directory where the corresponding data is located one by one, and separate the directory paths with `,` | String         | -             | input_name_1:/mnt/image/input_1_dir,input_name_2:input_2_dir |
| calibrate_size     | Mandatory | Calibration data size                                        | Integer        | -             | [1, 65535]                                                   |
| input_type         | Mandatory | Correction data file format type                             | String         | -             | IMAGE, BIN <br>IMAGE: image file data <br>BIN: binary `.bin` file data |
| image_to_format    | Optional  | Image format conversion                                      | String         | -             | RGB, GRAY, BGR                                               |
| normalize_mean     | Optional  | Normalized mean<br/>dst = (src - mean) / std                 | Vector         | -             | Channel 3: [mean_1, mean_2, mean_3] <br/>Channel 1: [mean_1] |
| normalize_std      | Optional  | Normalized standard deviation<br/>dst = (src - mean) / std   | Vector         | -             | Channel 3: [std_1, std_2, std_3] <br/>Channel 1: [std_1]     |
| resize_width       | Optional  | Resize width                                                 | Integer        | -             | [1, 65535]                                                   |
| resize_height      | Optional  | Resize height                                                | Integer        | -             | [1, 65535]                                                   |
| resize_method      | Optional  | Resize algorithm                                             | String         | -             | LINEAR, NEAREST, CUBIC<br/>LINEAR: Bilinear interpolation<br/>NEARST: Nearest neighbor interpolation<br/>CUBIC: Bicubic interpolation |
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

### Dynamic Quantization Parameters

The detailed description of the dynamic quantization parameter is as follows:

| Parameter  | Attribute | Function Description                                         | Parameter Type | Default Value | Value Range |
| ---------- | --------- | ------------------------------------------------------------ | -------------- | ------------- | ----------- |
| quant_strategy | Optional | the dynamic quantizaiton strategy | String  | ALWC   |  ALWC: Enable activation perlayer and weight perchannel quantization; <br/>ACWL: Enable activation perchannel and weight perlayer quantization. |

The dynamic quantization parameter configuration is as follows:

```ini
[dynamic_quant_param]
# If set to ALWC, it will enable activation perlayer and weight perchannel quantization. If set to ACWL, it will enable activation perchannel and weight perlayer quantization. Default value is ALWC.
quant_strategy=ACWL
```

## Quantization Debugging

Turning on the quantization debugging function enables you to get a statistical report of the data distribution, which can be used to assess the quantization error and assist in deciding whether the model (operator) is suitable for quantization. For full quantization, N data distribution statistics reports will be generated based on the number of corrected datasets provided, i.e., one report will be generated for each round; for weighted quantization, only 1 data distribution statistics report will be generated.

Setting the `debug_info_save_path` parameter will generate a Debug report in the `/home/workspace/mindspore/debug_info_save_path` folder:

```ini
[common_quant_param]
debug_info_save_path=/home/workspace/mindspore/debug_info_save_path
```

The quantization output summary report `output_summary.csv` contains information about the accuracy of the Tensor in the output layer of the whole graph, and the relevant fields are shown below:

| Type           | Name              |
| -------------- | ----------------- |
| Round       | Calibration training rounds          |
| TensorName       | name of the Tensor          |
| CosineSimilarity | Cosine similarity compared to the original data      |

The data distribution statistics report `round_*.csv` counts the distribution of the original data for each Tensor and the distribution of the data after inverse quantization of the quantized Tensor. The relevant fields of the data distribution statistics report are shown below:

| Type             | Name                                                     |
| ---------------- | -------------------------------------------------------- |
| NodeName         | Node name                                  |
| NodeType         | Node type                                                 |
| TensorName       | Tensor name                                                 |
| InOutFlag        | Tensor output, output type                                     |
| DataTypeFlag     | Data type, Origin for raw data, Dequant for inverse quantized data      |
| TensorTypeFlag   | Data classes such as inputs and outputs are represented as Activation, and constants are represented as Weight. |
| Min              | Minimum, 0% quantile point                                         |
| Q1               | 25% quantile point                                                |
| Median           | Median, 50% quantile point                                        |
| Q3               | 75% quantile point                                                |
| MAX              | Max value, 100% quantile point                                       |
| Mean             | Mean                                    |
| Var              | variance                                                     |
| Sparsity         | Sparsity                                                   |
| Clip             | truncation rate                                                   |
| CosineSimilarity | Cosine similarity compared to the original data                               |

The quantization parameter report `quant_param.csv` contains information about the quantization parameters of all quantization Tensor, the fields related to quantization parameters are shown below:

| Type           | Name              |
| -------------- | ----------------- |
| NodeName       | Node name            |
| NodeType       | Node type          |
| TensorName     | Tensor name          |
| ElementsNum    | Tensor data volume      |
| Dims           | Tensor dimension        |
| Scale          | Quantization parameter scale     |
| ZeroPoint      | Quantization parameter ZeroPoint |
| Bits           | Quantization bit        |
| CorrectionVar  | Error correction factor - variance |
| CorrectionMean | Error correction factor - mean |

> Since mixed bit quantization is non-standard quantization, this quantization parameter file may not exist.

### Partial Operators Skip Quantization

Quantization is to convert float32 operator to int8 operator. The current quantization strategy is the Node contained in a certain class of supportable operators will be quantized, but there is a part of the Node sensitivity is high, the quantization will trigger a large error, while some layers of quantization after the inference speed is much lower than the inference speed of float16. Supporting the specified layer without quantization can effectively improve the accuracy and inference speed.

The following is an example of `conv2d_1`, `add_8` and `concat_1` Node without quantization:

```ini
[common_quant_param]
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=FULL_QUANT
# Weight quantization supports the number of bits [0,16]. Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
# Full quantization supports 8bit
bit_num=8
# Set the name of the operator that skips the quantization, and use `,` to split between multiple operators.
skip_quant_node=conv2d_1,add_8,concat_1
```

### Suggestions for Use

1. By filtering `InOutFlag == Output && DataTypeFlag == Dequant`, you can filter out the output layer of all quantization operators, and determine the loss of precision of the operator by looking at the `CosineSimilarity` of the quantization output, the closer to 1 the smaller the loss.
2. For merge class operators such as Add and Concat, if the `min` and `max` distributions between different input Tensors have large differences, which is easy to trigger a large error, you can set `skip_quant_node` to unquantize them.
3. For operators with a high truncation rate `Clip`, you can set `skip_quant_node` to unquantize them.

## Classical Model Accuracy Results

### Weight Quantization

| Models                                                         | Test datasets                        | FP32 model accuracy | Weight quantization accuracy (8bit) |
| ------------------------------------------------------------ | --------------------------------- | ------------ | -------------------- |
| [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) | 77.60%       | 77.53%               |
| [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | [ImageNet](http://image-net.org/) | 70.96%       | 70.56%               |
| [Mobilenet_V2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) | [ImageNet](http://image-net.org/) | 71.56%       | 71.53%               |

All of the above results were measured on x86 environment.

### Full Quantization

| Model                                                        | Test Dataset                      | quant_method | FP32 Model Accuracy | Full Quantization Accuracy (8 bits) | Description                                                  |
| --------            | -------      | -----          | -----            | -----     | -----  |
| [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) | KL       | 77.60%              | 77.40%                              | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |
| [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | [ImageNet](http://image-net.org/) | KL       | 70.96%              | 70.31%                              | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |
| [Mobilenet_V2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz) | [ImageNet](http://image-net.org/) | MAX_MIN  | 71.56%              | 71.16%                              | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |

All of the above results were measured on x86 environment, CPU hardware backend.

### Dynamic Quantization

- tinybert_encoder model dynamic quantization vs. other quantization algorithms.

  | Model Type          | Runtime Mode  | Model Size(M) | RAM(K)     | Latency(ms) | Cos-Similarity | Compression Ratio | Memory compared to FP32 | Latency compared to FP32 |
  | ------------------- | ------------- | ------------- | ---------- | ----------- | -------------- | ----------------- | ----------------------- | ------------------------ |
  | FP32                | FP32          | 20            | 29,029     | 9.916       | 1              |                   |                         |                          |
  | FP32                | FP16          | 20            | 18,208     | 5.75        | 0.99999        | 1                 | -37.28%                 | -42.01%                  |
  | FP16                | FP16          | 12            | 18,105     | 5.924       | 0.99999        | 1.66667           | -37.63%                 | -40.26%                  |
  | Weight Quant(8 Bit) | FP16          | 5.3           | 19,324     | 5.764       | 0.99994        | 3.77358           | -33.43%                 | -41.87%                  |
  | **Dynamic Quant**   | **INT8+FP32** | **5.2**       | **15,709** | **4.517**   | **0.99668**    | **3.84615**       | **-45.89%**             | **-54.45%**              |

- tinybert_decoder model dynamic quantization vs. other quantization algorithms.

  | Model Type          | Runtime Mode  | Model Size(M) | RAM(K)     | Latency(ms) | Cos-Similarity | Compression Ratio | Memory compared to FP32 | Latency compared to FP32 |
  | ------------------- | ------------- | ------------- | ---------- | ----------- | -------------- | ----------------- | ----------------------- | ------------------------ |
  | FP32                | FP32          | 43            | 51,355     | 4.161       | 1              |                   |                         |                          |
  | FP32                | FP16          | 43            | 29,462     | 2.184       | 0.99999        | 1                 | -42.63%                 | -47.51%                  |
  | FP16                | FP16          | 22            | 29,440     | 2.264       | 0.99999        | 1.95455           | -42.67%                 | -45.59%                  |
  | Weight Quant(8 Bit) | FP16          | 12            | 32,285     | 2.307       | 0.99998        | 3.58333           | -37.13%                 | -44.56%                  |
  | **Dynamic Quant**   | **INT8+FP32** | **12**        | **22,181** | **2.074**   | **0.9993**     | **3.58333**       | **-56.81%**             | **-50.16%**              |

  All of the above results were measured on x86 environment.
