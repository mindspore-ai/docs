# benchmark_train

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/tools/benchmark_train_tool.md)

## 概述

与`benchmark`工具类似，MindSpore端侧训练为你提供了`benchmark_train`工具对训练后的模型进行基准测试。它不仅可以对模型前向推理执行耗时进行定量分析（性能），还可以通过指定模型输出进行可对比的误差分析（精度）。

## Linux环境使用说明

### 环境准备

使用`benchmark_train`工具，需要进行如下环境准备工作。

- 编译：`benchmark_train`工具代码在MindSpore源码的`mindspore/lite/tools/benchmark_train`目录中，参考构建文档中的[环境要求](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#环境要求)和[编译示例](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#模块构建编译选项)编译端侧训练框架。

- 配置环境变量：参考构建文档中的[编译输出](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#目录结构)，获得`benchmark_train`工具，并配置环境变量。假设您编译出的端侧训练框架压缩包所在完整路径为`/path/mindspore-lite-{version}-{os}-{arch}.tar.gz`，解压并配置环境变量的命令如下：

    ```bash
    cd /path
    tar xvf mindspore-lite-{version}-{os}-{arch}.tar.gz
    export LD_LIBRARY_PATH=/path/mindspore-lite-{version}-{os}-{arch}/runtime/lib:/path/mindspore-lite-{version}-{os}-{arch}/runtime/third_party/libjpeg-turbo/lib:/path/mindspore-lite-{version}-{os}-{arch}/runtime/third_party/hiai_ddk/lib:/path/mindspore-lite-{version}-{os}-{arch}/runtime/third_party/glog:${LD_LIBRARY_PATH}
    ```

benchmark_train工具所在完整路径为`/path/mindspore-lite-{version}-{os}-{arch}/tools/benchmark_train/benchmark_train`。

### 参数说明

使用编译好的`benchmark_train`工具进行模型的基准测试时，其命令格式如下所示。

```text
./benchmark_train [--modelFile=<MODELFILE>] [--accuracyThreshold=<ACCURACYTHRESHOLD>]
   [--expectedDataFile=<BENCHMARKDATAFILE>] [--warmUpLoopCount=<WARMUPLOOPCOUNT>]
   [--timeProfiling=<TIMEPROFILING>] [--help]
   [--inDataFile=<INDATAFILE>] [--epochs=<EPOCHS>]
   [--exportFile=<EXPORTFILE>]
```

下面提供详细的参数说明。

| 参数名            | 属性 | 功能描述                                                     | 参数类型                                                 | 默认值 | 取值范围 |
| ----------------- | ---- | ------------------------------------------------------------ | ------ | -------- | ---------------------------------- |
| `--modelFile=<MODELPATH>` | 必选 | 指定需要进行基准测试的MindSpore Lite模型文件路径。 | String | null  | -        |
| `--accuracyThreshold=<ACCURACYTHRESHOLD>` | 可选 | 指定准确度阈值。 | Float           | 0.5    | -        |
| `--expectedDataFile=<CALIBDATAPATH>` | 可选 | 指定标杆数据的文件路径。标杆数据作为该测试模型的对比输出，是该测试模型使用相同输入并由其他深度学习框架前向推理而来。 | String | null | - |
| `--help` | 可选 | 显示`benchmark_train`命令的帮助信息。 | - | - | - |
| `--warmUpLoopCount=<WARMUPLOOPCOUNT>` | 可选 | 指定测试模型在执行基准测试运行轮数前进行的模型预热推理次数。 | Integer | 3 | - |
| `--timeProfiling=<TIMEPROFILING>`  | 可选 | 性能验证时生效，指定是否使用TimeProfiler打印每个算子的耗时。 | Boolean | false | true, false |
| `--inDataFile=<INDATAPATH>` | 可选 | 指定测试模型输入数据的文件路径。如果未设置，则使用随机输入。 | String | null | - |
| `--epochs=<EPOCHS>` | 可选 | 指定循环训练的轮次，大于0时会执行训练EPOCHS次，并输出耗时数据。 | Integer | 0 | >=0 |
| `--exportFile=<EXPORTFILE>` | 可选 | 导出模型的路径。 | String | null | - |

### 使用示例

在使用`benchmark_train`工具进行模型基准测试时，可通过设置不同的参数，实现对其不同的测试功能。主要分为性能测试和精度测试。

#### 性能测试

`benchmark_train`工具进行的性能测试主要的测试指标为模型单次训练的耗时。在性能测试任务中，请设置`epochs`为大于1的数值，不需要设置`expectedDataFile`等标杆数据参数。但是，可以设置`timeProfiling`选项参数，控制是否输出在某设备上模型网络层的耗时，`timeProfiling`默认为false，例如：

```bash
./benchmark_train --modelFile=./models/test_benchmark.ms --epochs=10
```

这条命令使用随机输入，循环10次，其他参数使用默认值。该命令执行后会输出如下统计信息，该信息显示了测试模型在运行指定推理轮数后所统计出的单次推理最短耗时、单次推理最长耗时和平均推理耗时。

```text
Model = test_benchmark.ms, numThreads = 1, MinRunTime = 72.228996 ms, MaxRuntime = 73.094002 ms, AvgRunTime = 72.556000 ms
```

```bash
./benchmark_train --modelFile=./models/test_benchmark.ms --epochs=10 --timeProfiling=true
```

这条命令使用随机输入，并且输出模型网络层的耗时信息，其他参数使用默认值。该命令执行后，模型网络层的耗时会输出如下统计信息，在该例中，该统计信息按照`opName`和`optype`两种划分方式分别显示，`opName`表示算子名，`optype`表示算子类别，`avg`表示该算子的平均单次运行时间，`percent`表示该算子运行耗时占所有算子运行总耗时的比例，`calledTimess`表示该算子的运行次数，`opTotalTime`表示该算子运行指定次数的总耗时。最后，`total time`和`kernel cost`分别显示了该模型单次推理的平均耗时和模型推理中所有算子的平均耗时之和。

```text
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

#### 精度测试

`benchmark_train`工具进行的精度测试主要是通过设置标杆数据来对比验证MindSpore Lite训练后的模型输出的精确性。在精确度测试任务中，除了需要设置`modelFile`参数以外，还必须设置`inDataFile`、`expectedDataFile`参数。例如：

```bash
./benchmark_train --modelFile=./models/test_benchmark.ms --inDataFile=./input/test_benchmark.bin --accuracyThreshold=3 --expectedDataFile=./output/test_benchmark.out
```

这条命令指定了测试模型的输入数据、标杆数据（默认的输入及标杆数据类型均为float32），同时指定了模型推理程序在CPU上运行，并指定了准确度阈值为3%。该命令执行后会输出如下统计信息，该信息显示了测试模型的单条输入数据、输出节点的输出结果和平均偏差率以及所有节点的平均偏差率。

```text
InData0: 139.947 182.373 153.705 138.945 108.032 164.703 111.585 227.402 245.734 97.7776 201.89 134.868 144.851 236.027 18.1142 22.218 5.15569 212.318 198.43 221.853
================ Comparing Output data ================
Data of node age_out : 5.94584e-08 6.3317e-08 1.94726e-07 1.91809e-07 8.39805e-08 7.66035e-08 1.69285e-07 1.46246e-07 6.03796e-07 1.77631e-07 1.54343e-07 2.04623e-07 8.89609e-07 3.63487e-06 4.86876e-06 1.23939e-05 3.09981e-05 3.37098e-05 0.000107102 0.000213932 0.000533579 0.00062465 0.00296401 0.00993984 0.038227 0.0695085 0.162854 0.123199 0.24272 0.135048 0.169159 0.0221256 0.013892 0.00502971 0.00134921 0.00135701 0.000383242 0.000163475 0.000136294 9.77864e-05 8.00793e-05 5.73874e-05 3.53858e-05 2.18535e-05 2.04467e-05 1.85286e-05 1.05075e-05 9.34751e-06 6.12732e-06 4.55476e-06
Mean bias of node age_out : 0%
Mean bias of all nodes: 0%
=======================================================
```

### Dump功能

具体用法可参考`benchmark`工具的[Dump功能](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_tool.html#dump%E5%8A%9F%E8%83%BD)。