# Optimizing the Model (Quantization After Training)

`Windows` `Linux` `Model Converting` `Model Optimization` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/lite/source_en/use/post_training_quantization.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Converting a trained `float32` model into an `int8` model through quantization after training can reduce the model size and improve the inference performance. In MindSpore Lite, this function is integrated into the model conversion tool `conveter_lite`. You can add command line parameters to convert a model into a quantization model.

MindSpore Lite quantization after training is classified into two types:

1. Weight quantization: quantizes a weight of a model and compresses only the model size. `float32` inference is still performed during inference.
2. Full quantization: quantizes the weight and activation value of a model. The `int` operation is performed during inference to improve the model inference speed and reduce power consumption.

Data types and parameters required for the two types are different, but both can be set by using the conversion tool. For details about how to use the conversion tool `converter_lite`, see [Converting Training Models](https://www.mindspore.cn/tutorial/lite/en/r1.2/use/converter_tool.html). After the tool configuration is completed, you can enable quantization after training.

## Weight Quantization

Quantization of 1 to 16 bits is supported. A smaller number of quantization bits indicates a higher model compression ratio and a large accuracy loss. You can use the [Benchmark tool](https://www.mindspore.cn/tutorial/lite/en/r1.2/use/benchmark_tool.html) to evaluate the accuracy and determine the number of quantization bits. Generally, the average relative error (accuracyThreshold) is within 4% which is small. The following describes the usage and effect of weight quantization.

### Parameter Description

Generally, the weight quantization conversion command is as follows:

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --quantType=WeightQuant --bitNum=BitNumValue --quantWeightSize=ConvWeightQuantSizeThresholdValue --quantWeightChannel=ConvWeightQuantChannelThresholdValue
```

Parameters of this command are described as follows:

| Parameter | Attribute | Function Description | Parameter Type | Default Value | Value Range |
| -------- | ------- | -----       | -----    |----- | -----     |
| `--quantType=<QUANTTYPE>` | Mandatory |Set this parameter to WeightQuant to enable weight quantization. | String | - | WeightQuant |
| `--bitNum=<BITNUM>` | Optional | Number of bits for weight quantization. Currently, 1 to 16 bits are supported. | Integer | 8 | \[1, 16] |
| `--quantWeightSize=<QUANTWEIGHTSIZE>` | Optional | Set the threshold of the convolution kernel size for weight quantization. If the size of the convolution kernel is greater than the threshold, the weight is quantized. Recommended value: 500 | Integer | 0 | \[0, +∞) |
| `--quantWeightChannel=<QUANTWEIGHTCHANNEL>` | Optional | Set the threshold of the number of convolution channels for weight quantization. If the number of convolution channels is greater than the threshold, the weight is quantized. Recommended value: 16 | Integer | 16 | \[0, +∞)|

You can adjust the weight quantization parameters based on the model and your requirements.
> To ensure the accuracy of weight quantization, you are advised to set the value range of the `--bitNum` parameter to 8 bits to 16 bits.

### Procedure

1. Correctly build the `converter_lite` executable file. For details about how to obtain the `converter_lite` tool and configure environment variables, see [Building MindSpore Lite](https://www.mindspore.cn/tutorial/lite/en/r1.2/use/build.html).
2. Take the TensorFlow Lite model as an example. Run the following command to convert the weight quantization model:

    ```bash
    ./converter_lite --fmk=TFLITE --modelFile=Inception_v3.tflite --outputFile=Inception_v3.tflite --quantType=WeightQuant --bitNum=8 --quantWeightSize=0 --quantWeightChannel=0
    ```

3. After the preceding command is successfully executed, the quantization model `Inception_v3.tflite.ms` is obtained. The size of the quantization model usually decreases to one fourth of the FP32 model.

### Partial Model Accuracy Result

 | Model | Test Dataset | FP32 Model Accuracy | Weight Quantization Accuracy (8 bits) |
 | --------            | -------              | -----            | -----     |
 | [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) |  77.60%   |   77.53%  |
 | [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)      | [ImageNet](http://image-net.org/) |  70.96%  |  70.56% |
 | [Mobilenet_V2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz)      | [ImageNet](http://image-net.org/) | 71.56%  |  71.53%  |

> All the preceding results are obtained in the x86 environment.

## Full Quantization

In scenarios where the model running speed needs to be improved and the model running power consumption needs to be reduced, the full quantization after training can be used. The following describes the usage and effect of full quantization.

### Parameter Description

Generally, the full quantization conversion command is as follows:

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --quantType=PostTraining --bitNum=8 --configFile=config.cfg
```

Parameters of this command are described as follows:

| Parameter | Attribute | Function Description | Parameter Type | Default Value | Value Range |
| -------- | ------- | -----       | -----    |----- | -----     |
| `--quantType=<QUANTTYPE>` | Mandatory | Set this parameter to PostTraining to enable full quantization. | String | - | PostTraining |
| `--configFile=<CONFIGFILE>` | Mandatory | Path of a calibration dataset configuration file | String | - | - |
| `--bitNum=<BITNUM>` | Optional | Number of bits for full quantization. Currently, 1 to 8 bits are supported. | Integer | 8 | \[1, 8] |

To compute a quantization parameter of an activation value, you need to provide a calibration dataset. It is recommended that the calibration dataset be obtained from the actual inference scenario and can represent the actual input of a model. The number of data records is about 100.
The calibration dataset configuration file uses the `key=value` mode to define related parameters. The `key` to be configured is as follows:

| Parameter Name | Attribute | Function Description | Parameter Type | Default Value | Value Range |
| -------- | ------- | -----          | -----    | -----     |  ----- |
| image_path | Mandatory | Directory for storing a calibration dataset. If a model has multiple inputs, enter directories where the corresponding data is stored in sequence. Use commas (,) to separate them. | String | - | The directory stores the input data that can be directly used for inference. Since the current framework does not support data preprocessing, all data must be converted in advance to meet the input requirements of inference. |
| batch_count | Optional | Number of used inputs | Integer | 100 | (0, +∞) |
| method_x | Optional | Input and output data quantization algorithms at the network layer | String  |  KL  | KL, MAX_MIN, or RemovalOutlier.  <br> KL: quantizes and calibrates the data range based on [KL divergence](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf).  <br> MAX_MIN: data quantization parameter computed based on the maximum and minimum values.  <br> RemovalOutlier: removes the maximum and minimum values of data based on a certain proportion and then calculates the quantization parameters.  <br> If the calibration dataset is consistent with the input data during actual inference, MAX_MIN is recommended. If the noise of the calibration dataset is large, KL or RemovalOutlier is recommended.
| thread_num | Optional | Number of threads used when the calibration dataset is used to execute the inference process | Integer | 1 | (0, +∞) |
| bias_correction | Optional | Indicate whether to correct the quantization error. | Boolean | false | True or false. After this parameter is enabled, the accuracy of the converted model can be improved. You are advised to set this parameter to true. |

> For a multi-input model, different input data must be stored in different directories. In addition, names of all files in each directory must be sorted in ascending lexicographic order to ensure one-to-one mapping. For example, a model has two inputs input0 and input1, and there are two calibration datasets (batch_count=2). The data of input0 is stored in the /dir/input0/ directory. The input data files are data_1.bin and data_2.bin. The data of input1 is stored in the /dir/input1/ directory. The input data files are data_a.bin and data_b.bin. The (data_1.bin, data_a.bin) is regarded as a group of inputs and the (data_2.bin, data_b.bin) is regarded as another group of inputs.

### Procedure

1. Correctly build the `converter_lite` executable file.
2. Prepare a calibration dataset. Assume that the dataset is stored in the `/dir/images` directory. Configure the `config.cfg` file. The content is as follows:

    ```python
    image_path=/dir/images
    batch_count=100
    method_x=MAX_MIN
    thread_num=1
    bias_correction=true
    ```

   The calibration dataset can be a subset of the test dataset. Each file stored in the `/dir/images` directory must be pre-processed input data, and each file can be directly used as the input for inference.
3. Take the MindSpore model as an example. Run the following command to convert the full quantization model:

    ```bash
    ./converter_lite --fmk=MINDIR --modelFile=lenet.mindir --outputFile=lenet_quant --quantType=PostTraining --configFile=config.cfg
    ```

4. After the preceding command is successfully executed, the quantization model `lenet_quant.ms` is obtained. Generally, the size of the quantization model decreases to one fourth of the FP32 model.

### Partial Model Accuracy Result

 | Model | Test Dataset | method_x | FP32 Model Accuracy | Full Quantization Accuracy (8 bits) | Description |
 | --------            | -------      | -----          | -----            | -----     | -----  |
 | [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) | KL |    77.60%   |   77.40%   | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |
 | [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz) | [ImageNet](http://image-net.org/) | KL |    70.96%    |  70.31%  | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |
 | [Mobilenet_V2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz)      | [ImageNet](http://image-net.org/) | MAX_MIN |    71.56%    |  71.16%  | Randomly select 100 images from the ImageNet Validation dataset as a calibration dataset. |

> All the preceding results are obtained in the x86 environment, and `bias_correction=true` is set.
