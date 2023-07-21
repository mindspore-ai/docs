# 训练后量化

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.7/lite/tutorials/source_zh_cn/use/post_training_quantization.md)

## 概述

对于已经训练好的`float32`模型，通过训练后量化将模型转为`int8`模型，不仅能减小模型大小，而且能显著提高推理性能。在MindSpore端侧框架中，这部分功能集成在模型转换工具`conveter_lite`中，通过增加命令行参数，便能够转换得到量化后模型。
目前训练后量化属于alpha阶段（支持部分网络，不支持多输入模型），正在持续完善中。

```
./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --quantType=PostTraining --config_file=config.cfg
```
## 参数说明

|     参数  |  属性   |   功能描述   | 参数类型 | 默认值 | 取值范围  |
| -------- | ------- | -----       | -----    |----- | -----     | 
| --quantType   | 必选 | 设置为PostTraining，启用训练后量化 | String | - | 必须设置为PostTraining |
| --config_file | 必选 | 校准数据集配置文件路径             | String | - | -  |

为了计算激活值的量化参数，用户需要提供校准数据集。校准数据集最好来自真实推理场景，能表征模型的实际输入情况，数量在100个左右。
校准数据集配置文件采用`key=value`的方式定义相关参数，需要配置的`key`如下:

|   参数名  |  属性   |     功能描述    |  参数类型 |   默认值 | 取值范围  |
| -------- | ------- | -----          | -----    | -----     |  ----- |
| image_path  | 必选 | 存放校准数据集的目录  |      String                |   -   | 该目录存放可直接用于执行推理的输入数据。由于目前框架还不支持数据预处理，所有数据必须事先完成所需的转换，使得它们满足推理的输入要求。 |
| batch_count | 可选 | 使用的输入数目       | Integer  |  100  | 大于0  |
| method_x | 可选 | 网络层输入输出数据量化算法 | String  |  KL  | KL，MAX_MIN。 KL: 基于[KL散度](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)对数据范围作量化校准； MAX_MIN：基于最大值、最小值计算数据的量化参数。 在模型以及数据集比较较简单的情况下，推荐使用MAX_MIN      |
| thread_num | 可选 | 使用校准数据集执行推理流程时的线程数 | Integer  |  1  |  大于0   |

## 使用示例

1. 正确编译出`converter_lite`可执行文件。
2. 准备校准数据集，假设存放在`/dir/images`目录，编写配置文件`config.cfg`，内容如下：
    ```
    image_path=/dir/images
    batch_count=100
    method_x=MAX_MIN
    thread_num=1
    ```
   校准数据集可以选择测试数据集的子集，要求`/dir/images`目录下存放的每个文件均是预处理好的输入数据，每个文件都可以直接用于推理的输入。
3. 以MindSpore模型为例，执行带训练后量化的模型转换命令:
    ```
    ./converter_lite --fmk=MS --modelFile=lenet.ms --outputFile=lenet_quant --quantType=PostTraining --config_file=config.cfg
    ```
4. 上述命令执行成功后，便可得到量化后的模型lenet_quant.ms，通常量化后的模型大小会下降到FP32模型的1/4。

## 部分模型精度结果

 |  模型                |  测试数据集   | method_x      |  FP32模型精度    |  训练后量化精度 | 说明 |
 | --------            | -------      | -----          | -----            | -----     | -----  |
 | [Inception_V3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz) | [ImageNet](http://image-net.org/) | KL |    77.92%   |   77.95%   | 校准数据集随机选择ImageNet Validation数据集中的100张 |
 | [Mobilenet_V1_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz)      | [ImageNet](http://image-net.org/) | KL |    70.96%    |  70.69%  | 校准数据集随机选择ImageNet Validation数据集中的100张 |

> 以上所有结果均在x86环境上测得。
