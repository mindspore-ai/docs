# TimeProfiler工具

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.7/lite/tutorials/source_zh_cn/use/timeprofiler_tool.md)

## 概述

TimeProfiler工具可以对MindSpore Lite模型网络层的前向推理进行耗时分析，由C++语言编码实现。

## 环境准备

使用TimeProfiler工具，需要进行如下环境准备工作。

- 编译：TimeProfiler工具代码在MindSpore源码的`mindspore/lite/tools/time_profile`目录中，参考构建文档中的[环境要求](https://www.mindspore.cn/lite/tutorial/zh-CN/r0.7/build.html#id2)和[编译示例](https://www.mindspore.cn/lite/tutorial/zh-CN/r0.7/build.html#id4)执行编译。

- 运行：参考部署文档中的[编译输出](https://www.mindspore.cn/lite/tutorial/zh-CN/r0.7/build.html#id5)，获得`timeprofile`工具，并配置环境变量。

## 参数说明

使用编译好的TimeProfiler工具进行模型网络层耗时分析时，其命令格式如下所示。

```bash
./timeprofile --modelPath=<MODELPATH> [--help] [--loopCount=<LOOPCOUNT>] [--numThreads=<NUMTHREADS>] [--cpuBindMode=<CPUBINDMODE>] [--inDataPath=<INDATAPATH>] [--fp16Priority=<FP16PRIORITY>]
```

下面提供详细的参数说明。

| 参数名            | 属性 | 功能描述                                                     | 参数类型 | 默认值 | 取值范围 |
| ----------------- | ---- | ------------------------------------------------------------ | ------ | -------- | ---------------------------------- |
| `--help` | 可选 | 显示`timeprofiler`命令的帮助信息。 | - | - | - |
| `--modelPath=<MODELPATH> ` | 必选 | 指定需要进行耗时分析的MindSpore Lite模型的文件路径。 | String | null   | -        |
| `--loopCount=<LOOPCOUNT>` | 可选 | 指定TimeProfiler工具进行耗时分析时，模型推理的运行次数，其值为正整数。 | Integer | 100 | - |
| `--numThreads=<NUMTHREADS>` | 可选 | 指定模型推理程序运行的线程数。 | Integer | 4 | - |
| `--cpuBindMode=<CPUBINDMODE>` | 可选 | 指定模型推理程序运行时绑定的CPU核类型。 | Integer   | 1      | -1：表示中核<br>1：表示大核<br>0：表示不绑定 |
| `--inDataPath=<INDATAPATH>` | 可选 | 指定模型输入数据的文件路径。如果未设置，则使用随机输入。 | String | null | - |
| `--fp16Priority=<FP16PIORITY>` | 可选 | 指定是否优先使用float16算子。 | Bool | false | true, false |

## 使用示例

使用TimeProfiler对`test_timeprofiler.ms`模型的网络层进行耗时分析，并且设置模型推理循环运行次数为10，则其命令代码如下：

```bash
./timeprofile --modelPath=./models/test_timeprofiler.ms --loopCount=10
```

该条命令执行后，TimeProfiler工具会输出模型网络层运行耗时的相关统计信息。对于本例命令，输出的统计信息如下。其中统计信息按照`opName`和`optype`两种划分方式分别显示，`opName`表示算子名，`optype`表示算子类别，`avg`表示该算子的平均单次运行时间，`percent`表示该算子运行耗时占所有算子运行总耗时的比例，`calledTimess`表示该算子的运行次数，`opTotalTime`表示该算子运行指定次数的总耗时。最后，`total time`和`kernel cost`分别显示了该模型单次推理的平均耗时和模型推理中所有算子的平均耗时之和。

```
-----------------------------------------------------------------------------------------
opName                                                          avg(ms)         percent         calledTimess    opTotalTime
conv2d_1/convolution                                            2.264800        0.824012        10              22.648003
conv2d_2/convolution                                            0.223700        0.081390        10              2.237000
dense_1/BiasAdd                                                 0.007500        0.002729        10              0.075000
dense_1/MatMul                                                  0.126000        0.045843        10              1.260000
dense_1/Relu                                                    0.006900        0.002510        10              0.069000
max_pooling2d_1/MaxPool                                         0.035100        0.012771        10              0.351000
max_pooling2d_2/MaxPool                                         0.014300        0.005203        10              0.143000
max_pooling2d_2/MaxPool_nchw2nhwc_reshape_1/Reshape_0           0.006500        0.002365        10              0.065000
max_pooling2d_2/MaxPool_nchw2nhwc_reshape_1/Shape_0             0.010900        0.003966        10              0.109000
output/BiasAdd                                                  0.005300        0.001928        10              0.053000
output/MatMul                                                   0.011400        0.004148        10              0.114000
output/Softmax                                                  0.013300        0.004839        10              0.133000
reshape_1/Reshape                                               0.000900        0.000327        10              0.009000
reshape_1/Reshape/shape                                         0.009900        0.003602        10              0.099000
reshape_1/Shape                                                 0.002300        0.000837        10              0.023000
reshape_1/strided_slice                                         0.009700        0.003529        10              0.097000
-----------------------------------------------------------------------------------------
opType          avg(ms)         percent         calledTimess    opTotalTime
Activation      0.006900        0.002510        10              0.069000
BiasAdd         0.012800        0.004657        20              0.128000
Conv2D          2.488500        0.905401        20              24.885004
MatMul          0.137400        0.049991        20              1.374000
Nchw2Nhwc       0.017400        0.006331        20              0.174000
Pooling         0.049400        0.017973        20              0.494000
Reshape         0.000900        0.000327        10              0.009000
Shape           0.002300        0.000837        10              0.023000
SoftMax         0.013300        0.004839        10              0.133000
Stack           0.009900        0.003602        10              0.099000
StridedSlice    0.009700        0.003529        10              0.097000

total time :     2.90800 ms,    kernel cost : 2.74851 ms

-----------------------------------------------------------------------------------------
```