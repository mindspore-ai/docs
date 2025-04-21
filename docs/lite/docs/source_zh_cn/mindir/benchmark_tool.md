# benchmark

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/mindir/benchmark_tool.md)

## 概述

转换模型后执行推理前，可以使用Benchmark工具对MindSpore Lite云侧推理模型进行基准测试。它不仅可以对MindSpore Lite云侧推理模型前向推理执行耗时进行定量分析（性能），还可以通过指定模型输出进行可对比的误差分析（精度）。

## Linux环境使用说明

### 环境准备

使用Benchmark工具，需要进行如下环境准备工作。

- 编译：Benchmark工具代码在MindSpore源码的`mindspore/lite/tools/benchmark`目录中，参考构建文档中的[环境要求](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/build.html#环境准备)和[编译示例](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/build.html#编译示例)执行编译。

- 运行：参考构建文档中的[编译输出](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/build.html#目录结构)，从编译出来的包中获得`benchmark`工具。

- 将推理需要的动态链接库加入环境变量LD_LIBRARY_PATH。

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/runtime/lib:${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ```

    其中${PACKAGE_ROOT_PATH}是编译得到的包解压后的根目录。

- 如果基于Ascend进行基准测试，使用如下命令切换：

    ```bash
    export ASCEND_DEVICE_ID=$RANKK_ID
    ```

### 参数说明

使用编译好的Benchmark工具进行模型的基准测试时，其命令格式如下所示。

```text
./benchmark [--modelFile=<MODELFILE>] [--accuracyThreshold=<ACCURACYTHRESHOLD>]
   [--cosineDistanceThreshold=<COSINEDISTANCETHRESHOLD>]
   [--benchmarkDataFile=<BENCHMARKDATAFILE>] [--benchmarkDataType=<BENCHMARKDATATYPE>]
   [--cpuBindMode=<CPUBINDMODE>] [--device=<DEVICE>] [--help]
   [--inDataFile=<INDATAFILE>] [--loopCount=<LOOPCOUNT>]
   [--numThreads=<NUMTHREADS>] [--warmUpLoopCount=<WARMUPLOOPCOUNT>]
   [--enableFp16=<ENABLEFP16>] [--timeProfiling=<TIMEPROFILING>]
   [--inputShapes=<INPUTSHAPES>] [--perfProfiling=<PERFPROFILING>]
   [--perfEvent=<PERFEVENT>]
```

下面提供详细的参数说明。

| 参数名            | 属性 | 功能描述                                                     | 参数类型                                                 | 默认值 | 取值范围 |  备注  |
| ----------------- | ---- | ------------------------------------------------------------ | ------ | -------- | ------------------ | ---- |
| `--modelFile=<MODELPATH>` | 必选 | 指定需要进行基准测试的MindSpore Lite模型文件路径。 | String | null  | -        |
| `--accuracyThreshold=<ACCURACYTHRESHOLD>` | 可选 | 指定准确度阈值。 | Float           | 0.5    | -        |
| `--benchmarkDataFile=<CALIBDATAPATH>` | 可选 | 指定标杆数据的文件路径。标杆数据作为该测试模型的对比输出，是该测试模型使用相同输入并由其他深度学习框架前向推理而来。 | String | null | - |
| `--benchmarkDataType=<CALIBDATATYPE>` | 可选 | 指定标杆数据类型。 | String | FLOAT | FLOAT、INT32、INT8、UINT8 |
| `--device=<DEVICE>` | 可选 | 指定模型推理程序运行的设备类型。 | String | CPU | CPU、GPU、NPU、Ascend |
| `--help` | 可选 | 显示`benchmark`命令的帮助信息。 | - | - | - |
| `--inDataFile=<INDATAPATH>` | 可选 | 指定测试模型输入数据的文件路径。如果未设置，则使用随机输入。 | String | null | - |
| `--loopCount=<LOOPCOUNT>` | 可选 | 指定Benchmark工具进行基准测试时，测试模型的前向推理运行次数，其值为正整数。 | Integer | 10 | - |
| `--numThreads=<NUMTHREADS>` | 可选 | 指定模型推理程序运行的线程数。 | Integer | 2 | - |
| `--warmUpLoopCount=<WARMUPLOOPCOUNT>` | 可选 | 指定测试模型在执行基准测试运行轮数前进行的模型预热推理次数。 | Integer | 3 | - |
| `--inputShapes=<INPUTSHAPES>` | 可选 | 指定输入维度，维度应该按照原始模型格式。维度值之间用‘,'隔开，多个输入的维度之间用‘:’隔开 | String | Null | - |
| `--cosineDistanceThreshold=<COSINEDISTANCETHRESHOLD>` | 可选 | 指定余弦距离阈值，只有指定该参数并且其值大于-1时，才会计算余弦距离。 | Float           | -1.1    | -        | 暂不支持 |
| `--cpuBindMode=<CPUBINDMODE>` | 可选 | 指定模型推理程序运行时绑定的CPU核类型。 | Integer | 1      | 2：表示中核<br/>1：表示大核<br/>0：表示不绑定 | 暂不支持 |
| `--enableFp16=<FP16PIORITY>` | 可选 | 指定是否优先使用float16算子。 | Boolean | false | true、false | 暂不支持 |
| `--timeProfiling=<TIMEPROFILING>`  | 可选 | 性能验证时生效，指定是否使用TimeProfiler打印每个算子的耗时。 | Boolean | false | true、false | 暂不支持 |
| `--perfEvent=<PERFEVENT>` | 可选 | CPU性能验证时生效，指定PerfProfiler打印的CPU性能参数的具体内容，指定为CYCLE时，会打印算子的CPU周期数和指令条数；指定为CACHE时，会打印算子的缓存读取次数和缓存未命中次数；指定为STALL时，会打印CPU前端等待周期数和后端等待周期数。 | String | CYCLE | CYCLE/CACHE/STALL | 暂不支持 |
| `--decryptKey=<DECRYPTKEY>` | 可选 | 用于解密文件的密钥，以十六进制字符表示。仅支持 AES-GCM，密钥长度仅支持16Byte。 | String | null | 注意密钥为十六进制表示的字符串，如密钥定义为`b'0123456789ABCDEF'`对应的十六进制表示为`30313233343536373839414243444546`，Linux平台用户可以使用`xxd`工具对字节表示的密钥进行十六进制表达转换。 | 暂不支持 |
| `--cryptoLibPath=<CRYPTOLIBPATH>` | 可选 | OpenSSL加密库crypto的路径 | String | null | - | 暂不支持 |

### 使用示例

对于不同的MindSpore Lite云侧推理模型，在使用Benchmark工具对其进行基准测试时，可通过设置不同的参数，实现对其不同的测试功能。主要分为性能测试和精度测试。

#### 性能测试

Benchmark工具进行的性能测试主要的测试指标为模型单次前向推理的耗时。在性能测试任务中，不需要设置`benchmarkDataFile`等标杆数据参数。例如：

```bash
./benchmark --modelFile=/path/to/model.mindir
```

这条命令使用随机输入，其他参数使用默认值。该命令执行后会输出如下统计信息，该信息显示了测试模型在运行指定推理轮数后所统计出的单次推理最短耗时、单次推理最长耗时和平均推理耗时。

```text
Model = model.mindir, numThreads = 2, MinRunTime = 72.228996 ms, MaxRuntime = 73.094002 ms, AvgRunTime = 72.556000 ms
```

#### 精度测试

Benchmark工具进行的精度测试主要是通过设置标杆数据来对比验证MindSpore Lite云侧推理模型输出的精确性。在精确度测试任务中，除了需要设置`modelFile`参数以外，还必须设置`benchmarkDataFile`参数。例如：

```bash
./benchmark --modelFile=/path/to/model.mindir --inDataFile=/path/to/input.bin --device=CPU --accuracyThreshold=3 --benchmarkDataFile=/path/to/output.out
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

如果需要指定输入数据的维度（例如输入维度为1，32，32，1），使用如下命令：

```bash
./benchmark --modelFile=/path/to/model.mindir --inDataFile=/path/to/input.bin --inputShapes=1,32,32,1 --device=CPU --accuracyThreshold=3 --benchmarkDataFile=/path/to/output.out
```

如果输入的模型是加密模型，需要同时配置`decryptKey`和`cryptoLibPath`对模型解密后进行推理，使用如下命令：

```bash
./benchmark --modelFile=/path/to/encry_model.mindir --decryptKey=30313233343536373839414243444546 --cryptoLibPath=/root/anaconda3/bin/openssl
```