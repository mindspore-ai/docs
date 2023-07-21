# 优化模型(训练后量化)

`Windows` `Linux` `模型转换` `模型调优` `中级` `高级`

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/lite/source_zh_cn/use/post_training_quantization.md)

## 概述

对于已经训练好的`float32`模型，通过训练后量化将其转为`int8`，不仅能减小模型大小，而且能显著提高推理性能。在MindSpore Lite中，这部分功能集成在模型转换工具`conveter_lite`内，通过增加命令行参数，便能够转换得到量化后模型。

MindSpore Lite训练后量化分为两类：

1. 权重量化：对模型的权值进行量化，仅压缩模型大小，推理时仍然执行`float32`推理；
2. 全量化：对模型的权值、激活值等统一进行量化，推理时执行`int`运算，能提升模型推理速度、降低功耗。

训练后量化在两种情况下所需的数据类型和参数设定不同，但均可通过转换工具设定。有关转换工具`converter_lite`的使用方法可参考[转换为MindSpore Lite模型](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/converter_tool.html)。在此基础之上进行配置，启用训练后量化。

## 权重量化

支持1~16之间的任意比特量化，量化比特数越低，模型压缩率越大，但是精度损失通常也比较大。可以结合使用[Benchmark工具](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/benchmark_tool.html)进行精度评估，确定合适的量化比特数；通常平均相对误差(accuracyThreshold)满足4%以内，精度误差是比较小的。下面对权重量化的使用方式和效果进行阐述。

### 参数说明

权重量化转换命令的一般形式为：

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --quantType=WeightQuant --bitNum=BitNumValue --quantWeightSize=ConvWeightQuantSizeThresholdValue --quantWeightChannel=ConvWeightQuantChannelThresholdValue
```

下面对此命令的量化相关参数进行说明：

|     参数  |  属性   |   功能描述   | 参数类型 | 默认值 | 取值范围  |
| -------- | ------- | -----       | -----    |----- | -----     |
| `--quantType=<QUANTTYPE>`   | 必选 | 设置为WeightQuant，启用权重量化 | String | - | 必须设置为WeightQuant |
| `--bitNum=<BITNUM>` | 可选 | 设定权重量化的比特数，目前支持1bit～16bit量化 | Integer | 8 | \[1，16] |
| `--quantWeightSize=<QUANTWEIGHTSIZE>` | 可选 | 设定参与权重量化的卷积核尺寸阈值，若卷积核尺寸大于该值，则对此权重进行量化；建议设置为500 | Integer | 0 | \[0，+∞） |
| `--quantWeightChannel=<QUANTWEIGHTCHANNEL>` | 可选 | 设定参与权重量化的卷积通道数阈值，若卷积通道数大于该值，则对此权重进行量化；建议设置为16 | Integer | 16 | \[0，+∞） |

用户可根据模型及自身需要对权重量化的参数作出调整。
> 为保证权重量化的精度，建议`--bitNum`参数设定范围为8bit～16bit。

### 使用步骤

1. 正确编译出`converter_lite`可执行文件。该部分可参考构建文档[编译MindSpore Lite](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.1/use/build.html)，获得`converter_lite`工具，并配置环境变量。
2. 以TensorFlow Lite模型为例，执行权重量化模型转换命令:

    ```bash
    ./converter_lite --fmk=TFLITE --modelFile=Inception_v3.tflite --outputFile=Inception_v3.tflite --quantType=WeightQuant --bitNum=8 --quantWeightSize=0 --quantWeightChannel=0
    ```

3. 上述命令执行成功后，便可得到量化后的模型`Inception_v3.tflite.ms`，量化后的模型大小通常会下降到FP32模型的1/4。

### 部分模型精度结果

 |  模型                |  测试数据集        |  FP32模型精度    |  权重量化精度（8bit） |
 | --------            | -------              | -----            | -----     |
 | [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) |  77.60%   |   77.53%  |
 | [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)      | [ImageNet](http://image-net.org/) |  70.96%  |  70.56% |
 | [Mobilenet_V2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz)      | [ImageNet](http://image-net.org/) | 71.56%  |  71.53%  |

> 以上所有结果均在x86环境上测得。

## 全量化

针对需要提升模型运行速度、降低模型运行功耗的场景，可以使用训练后全量化功能。下面对全量化的使用方式和效果进行阐述。

### 参数说明

全量化转换命令的一般形式为：

```bash
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --quantType=PostTraining --bitNum=8 --configFile=config.cfg
```

下面对此命令的量化相关参数进行说明：

|     参数  |  属性   |   功能描述   | 参数类型 | 默认值 | 取值范围  |
| -------- | ------- | -----       | -----    |----- | -----     |
| `--quantType=<QUANTTYPE>`   | 必选 | 设置为PostTraining，启用全量化 | String | - | 必须设置为PostTraining |
| `--configFile=<CONFIGFILE>` | 必选 | 校准数据集配置文件路径  | String | - | -  |
| `--bitNum=<BITNUM>` | 可选 | 设定全量化的比特数，目前支持1bit～8bit量化 | Integer | 8 | \[1，8] |

为了计算激活值的量化参数，用户需要提供校准数据集。校准数据集最好来自真实推理场景，能表征模型的实际输入情况，数量在100个左右。
校准数据集配置文件采用`key=value`的方式定义相关参数，需要配置的`key`如下:

|   参数名  |  属性   |     功能描述    |  参数类型 |   默认值 | 取值范围  |
| -------- | ------- | -----          | -----    | -----     |  ----- |
| image_path  | 必选 | 存放校准数据集的目录；如果模型有多个输入，请依次填写对应的数据所在目录，目录路径间请用`,`隔开 |      String                |   -   | 该目录存放可直接用于执行推理的输入数据。由于目前框架还不支持数据预处理，所有数据必须事先完成所需的转换，使得它们满足推理的输入要求 |
| batch_count | 可选 | 使用的输入数目       | Integer  |  100  | （0，+∞） |
| method_x | 可选 | 网络层输入输出数据量化算法 | String  |  KL  | KL、MAX_MIN、RemovalOutlier。 <br> KL：基于[KL散度](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)对数据范围作量化校准。 <br> MAX_MIN：基于最大值、最小值计算数据的量化参数。 <br> RemovalOutlier：按照一定比例剔除数据的极大极小值，再计算量化参数。 <br> 在校准数据集与实际推理时的输入数据相吻合的情况下，推荐使用MAX_MIN；而在校准数据集噪声比较大的情况下，推荐使用KL或者RemovalOutlier      |
| thread_num | 可选 | 使用校准数据集执行推理流程时的线程数 | Integer  |  1  |  （0，+∞）   |
| bias_correction | 可选 | 是否对量化误差进行校正 | Boolean  |  false  |  true、flase。使能后，能提升转换后的模型精度，建议设置为true |

> 对于多输入模型，要求不同输入数据分别存放在各自不同的目录，同时各自目录中的所有文件的文件名按照字典序递增排序后，能够一一对应。例如：模型有两个输入input0、input1，校准数据集共2组（batch_count=2）；input0的对应数据存放在/dir/input0/目录，输入数据文件名为：data_1.bin、data_2.bin；input1的对应数据存放在/dir/input1/目录，输入数据文件名为：data_a.bin、data_b.bin，则认为(data_1.bin, data_a.bin)构成一组输入，（data_2.bin, data_b.bin）构成另一组输入。

### 使用步骤

1. 正确编译出`converter_lite`可执行文件。
2. 准备校准数据集，假设存放在`/dir/images`目录，编写配置文件`config.cfg`，内容如下：

    ```python
    image_path=/dir/images
    batch_count=100
    method_x=MAX_MIN
    thread_num=1
    bias_correction=true
    ```

   校准数据集可以选择测试数据集的子集，要求`/dir/images`目录下存放的每个文件均是预处理好的输入数据，每个文件都可以直接用于推理的输入。
3. 以MindSpore模型为例，执行全量化的模型转换命令:

    ```bash
    ./converter_lite --fmk=MINDIR --modelFile=lenet.mindir --outputFile=lenet_quant --quantType=PostTraining --configFile=config.cfg
    ```

4. 上述命令执行成功后，便可得到量化后的模型`lenet_quant.ms`，通常量化后的模型大小会下降到FP32模型的1/4。

### 部分模型精度结果

 |  模型                |  测试数据集   | method_x      |  FP32模型精度    |  全量化精度（8bit） | 说明 |
 | --------            | -------      | -----          | -----            | -----     | -----  |
 | [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) | KL |    77.60%   |   77.40%   | 校准数据集随机选择ImageNet Validation数据集中的100张 |
 | [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)      | [ImageNet](http://image-net.org/) | KL |    70.96%    |  70.31%  | 校准数据集随机选择ImageNet Validation数据集中的100张 |
 | [Mobilenet_V2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz)      | [ImageNet](http://image-net.org/) | MAX_MIN |    71.56%    |  71.16%  | 校准数据集随机选择ImageNet Validation数据集中的100张 |

> 以上所有结果均在x86环境上测得，均设置`bias_correction=true`。
