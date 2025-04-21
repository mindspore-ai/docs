# benchmark

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/tools/benchmark_tool.md)

## 概述

转换模型后执行推理前，你可以使用Benchmark工具对MindSpore Lite模型进行基准测试。它不仅可以对MindSpore Lite模型前向推理执行耗时进行定量分析（性能），还可以通过指定模型输出进行可对比的误差分析（精度）。

## Linux环境使用说明

### 环境准备

使用Benchmark工具，需要进行如下环境准备工作。

- 编译：Benchmark工具代码在MindSpore源码的`mindspore/lite/tools/benchmark`目录中，参考构建文档中的[环境要求](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#环境要求)和[编译示例](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#编译示例)执行编译。

- 运行：参考构建文档中的[编译输出](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#编译选项)，获得`benchmark`工具。

- 将推理需要的动态链接库加入环境变量LD_LIBRARY_PATH。

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/runtime/lib:${LD_LIBRARY_PATH}
    ```

    其中${PACKAGE_ROOT_PATH}是编译得到的包解压后的根目录。

- 如果基于Ascend进行基准测试，使用如下命令切换：

    ```bash
    export ASCEND_DEVICE_ID=$RANK_ID
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

| 参数名            | 属性 | 功能描述                                                     | 参数类型                                                 | 默认值 | 取值范围 |
| ----------------- | ---- | ------------------------------------------------------------ | ------ | -------- | ------------------------------- |
| `--modelFile=<MODELPATH>` | 必选 | 指定需要进行基准测试的MindSpore Lite模型文件路径。 | String | null  | -        |
| `--accuracyThreshold=<ACCURACYTHRESHOLD>` | 可选 | 指定准确度阈值。 | Float           | 0.5    | -        |
| `--cosineDistanceThreshold=<COSINEDISTANCETHRESHOLD>` | 可选 | 指定余弦距离阈值，只有指定该参数并且其值大于-1时，才会计算余弦距离。 | Float           | -1.1    | -        |
| `--benchmarkDataFile=<CALIBDATAPATH>` | 可选 | 指定标杆数据的文件路径。标杆数据作为该测试模型的对比输出，是该测试模型使用相同输入并由其他深度学习框架前向推理而来。 | String | null | - |
| `--benchmarkDataType=<CALIBDATATYPE>` | 可选 | 指定标杆数据类型。 | String | FLOAT | FLOAT、INT32、INT8、UINT8 |
| `--cpuBindMode=<CPUBINDMODE>` | 可选 | 指定模型推理程序运行时绑定的CPU核类型。 | Integer | 1      | 2：表示中核<br/>1：表示大核<br/>0：表示不绑定 |
| `--device=<DEVICE>` | 可选 | 指定模型推理程序运行的设备类型。 | String | CPU | CPU、GPU、NPU、Ascend |
| `--help` | 可选 | 显示`benchmark`命令的帮助信息。 | - | - | - |
| `--inDataFile=<INDATAPATH>` | 可选 | 指定测试模型输入数据的文件路径。如果未设置，则使用随机输入。 | String | null | - |
| `--loopCount=<LOOPCOUNT>` | 可选 | 指定Benchmark工具进行基准测试时，测试模型的前向推理运行次数，其值为正整数。 | Integer | 10 | - |
| `--numThreads=<NUMTHREADS>` | 可选 | 指定模型推理程序运行的线程数。 | Integer | 2 | - |
| `--warmUpLoopCount=<WARMUPLOOPCOUNT>` | 可选 | 指定测试模型在执行基准测试运行轮数前进行的模型预热推理次数。 | Integer | 3 | - |
| `--enableFp16=<FP16PIORITY>` | 可选 | 指定是否优先使用float16算子。 | Boolean | false | true, false |
| `--timeProfiling=<TIMEPROFILING>`  | 可选 | 性能验证时生效，指定是否使用TimeProfiler打印每个算子的耗时。 | Boolean | false | true, false |
| `--inputShapes=<INPUTSHAPES>` | 可选 | 指定输入维度，维度应该按照NHWC格式输入。维度值之间用‘,'隔开，多个输入的维度之间用‘:’隔开 | String | Null | - |
| `--perfProfiling=<PERFPROFILING>` | 可选 | CPU性能验证时生效，指定是否使用PerfProfiler打印每个算子的CPU性能，当timeProfiling为true时无效。目前仅支持aarch64 CPU。 | Boolean | false | true, false |
| `--perfEvent=<PERFEVENT>` | 可选 | CPU性能验证时生效，指定PerfProfiler打印的CPU性能参数的具体内容，指定为CYCLE时，会打印算子的CPU周期数和指令条数；指定为CACHE时，会打印算子的缓存读取次数和缓存未命中次数；指定为STALL时，会打印CPU前端等待周期数和后端等待周期数。 | String | CYCLE | CYCLE/CACHE/STALL |
| `--decryptKey=<DECRYPTKEY>` | 可选 | 用于解密文件的密钥，以十六进制字符表示。仅支持 AES-GCM，密钥长度仅支持16Byte。 | String | null | 注意密钥为十六进制表示的字符串，如密钥定义为`b'0123456789ABCDEF'`对应的十六进制表示为`30313233343536373839414243444546`，Linux平台用户可以使用`xxd`工具对字节表示的密钥进行十六进制表达转换。 |
| `--cryptoLibPath=<CRYPTOLIBPATH>` | 可选 | OpenSSL加密库crypto的路径 | String | null | - |

### 使用示例

对于不同的MindSpore Lite模型，在使用Benchmark工具对其进行基准测试时，可通过设置不同的参数，实现对其不同的测试功能。主要分为性能测试和精度测试。

#### 性能测试

Benchmark工具进行的性能测试主要的测试指标为模型单次前向推理的耗时。在性能测试任务中，不需要设置`benchmarkDataFile`等标杆数据参数。但是，可以设置`timeProfiling`选项参数，控制是否输出在某设备上模型网络层的耗时，`timeProfiling`默认为false，例如：

```bash
./benchmark --modelFile=/path/to/model.ms
```

这条命令使用随机输入，其他参数使用默认值。该命令执行后会输出如下统计信息，该信息显示了测试模型在运行指定推理轮数后所统计出的单次推理最短耗时、单次推理最长耗时和平均推理耗时。

```text
Model = model.ms, numThreads = 2, MinRunTime = 72.228996 ms, MaxRuntime = 73.094002 ms, AvgRunTime = 72.556000 ms
```

```bash
./benchmark --modelFile=/path/to/model.ms --timeProfiling=true
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

Benchmark工具进行的精度测试主要是通过设置[标杆数据（input.bin和output.out）](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_golden_data.html) 来对比验证MindSpore Lite模型输出的精确性。在精确度测试任务中，除了需要设置`modelFile`参数以外，还必须设置`benchmarkDataFile`参数。例如：

```bash
./benchmark --modelFile=/path/to/model.ms --inDataFile=/path/to/input.bin --device=CPU --accuracyThreshold=3 --benchmarkDataFile=/path/to/output.out
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
./benchmark --modelFile=/path/to/model.ms --inDataFile=/path/to/input.bin --inputShapes=1,32,32,1 --device=CPU --accuracyThreshold=3 --benchmarkDataFile=/path/to/output.out
```

#### CPU性能测试

Benchmark工具进行的CPU性能测试主要的测试指标为模型单次前向推理CPU性能参数(目前只支持aarch64 CPU)，包括周期数和指令数、缓存读取次数和缓存未命中次数、CPU前端和后端等待时间。在CPU性能测试任务中，不需要设置`benchmarkDataFile`等标杆数据参数。但是，可以设置`perfProfiling`与`perfEvent`选项参数，控制输出在某设备上模型网络层的哪些CPU性能参数，`perfProfiling`默认为false，`perfEvent`默认为`CYCLE`(CPU周期数和指令数)。由于多线程的读数波动较大，建议设置线程数为1。使用方法如下：

```bash
./benchmark --modelFile=/path/to/model.ms --perfProfiling=true --numThreads=1
```

这条命令使用随机输入，并且输出模型网络层的周期数/指令数信息，其他参数使用默认值。该命令执行后，会输出如下CPU性能参数统计信息，在该例中，该统计信息按照`opName`和`optype`两种划分方式分别显示，`opName`表示算子名，`optype`表示算子类别，`cycles(k)`表示该算子的平均CPU周期数（以k为单位，受CPU频率影响），`cycles(%)`表示该算子CPU周期数占所有算子CPU周期数的比例，`ins(k)`表示该算子的指令数（以k为单位），`ins(%)`表示该算子的指令数占所有算子指令数的比例。最后会显示当前模型、线程数、最小运行时间、最大运行时间、平均运行时间用做参考。

```text
-----------------------------------------------------------------------------------------
opName                                                   cycles(k)       cycles(%)       ins(k)          ins(%)
Add_Plus214_Output_0                                     1.53            0.006572        1.27            0.002148
Conv_Convolution110_Output_0                             91.12           0.390141        217.58          0.369177
Conv_COnvolution28_Output_0                              114.61          0.490704        306.28          0.519680
Matmul_Times212_Output_0                                 8.75            0.037460        15.55           0.026385
MaxPool_Pooling160_Output_0                              3.24            0.013873        8.70            0.014767
MaxPool_Pooling66_Output_0                               11.63           0.049780        35.17           0.059671
Reshape_Pooling160_Output_0_reshape0                     0.91            0.003899        1.58            0.002677
nhwc2nchw_MaxPool_Pooling160_Output_0_post8_0            1.77            0.007571        3.25            0.005508
-----------------------------------------------------------------------------------------
opType          cycles(k)       cycles(%)       ins(k)          ins(%)
Add             1.53            0.006572        1.27            0.002148
Conv2D          205.73          0.880845        523.85          0.888856
MatMul          8.75            0.037460        15.55           0.026385
Nhwc2nchw       1.77            0.007571        3.25            0.005508
Pooling         14.87           0.063654        43.87           0.074437
Reshape         0.91            0.003839        1.58            0.002677

Model = model.ms, NumThreads = 1, MinRunTime = 0.104000 ms, MaxRunTime = 0.179000 ms, AvgRunTime = 0.116000 ms

-----------------------------------------------------------------------------------------
```

当`perfEvent`参数被指定为`CACHE`时，列标题会变为`cache ref(k)`/`cache ref(%)`/`miss(k)`/`miss(%)`，分别代表算子缓存读取次数/缓存读取占比/缓存未命中次数/缓存未命中次数占比；当`perfEvent`参数被指定为`STALL`时，列标题会变为`frontend(k)`/`frontend(%)`/`backend(k)`/`backend(%)`，分别代表CPU前端等待时间/CPU前端等待时间占比/CPU后端等待时间/CPU后端等待时间数占比。使用方法如下：

```bash
./benchmark --modelFile=/path/to/model.ms --perfProfiling=true --perfEvent="CACHE"
```

```bash
./benchmark --modelFile=/path/to/model.ms --perfProfiling=true --perfEvent="STALL"
```

### Dump功能

Benchmark工具提供Dump功能（目前仅支持`CPU`和移动端`GPU`算子），将模型中的算子的输入输出数据保存到磁盘文件中，可用于定位模型推理过程中精度异常的问题。

#### Dump操作步骤

1. 创建json格式的配置文件，JSON文件的名称和位置可以自定义设置。

    ```json
    {
        "common_dump_settings": {
            "dump_mode": 1,
            "path": "/absolute_path",
            "net_name": "ResNet50",
            "input_output": 0,
            "kernels": ["Default/Conv-op12", "Default/Conv-op13"]
        }
    }
    ```

    - `dump_mode`：设置成0，表示Dump出该网络中的所有算子数据；设置成1，表示Dump`"kernels"`里面指定的算子数据。
    - `path`：Dump保存数据的绝对路径。
    - `net_name`：自定义的网络名称，例如："ResNet50"，未指定该字段的话，默认值为"default"。
    - `input_output`：设置成0，表示Dump出算子的输入和算子的输出；设置成1，表示Dump出算子的输入；设置成2，表示Dump出算子的输出。
    - `kernels`：算子的名称列表。如果未指定此字段或者此字段的值设置为[]，`"dump_mode"`须设置为0；否则`"dump_mode"`的值须设置为1。

2. 设置Dump环境变量，指定Dump的json配置文件。

   ```bash
   export MINDSPORE_DUMP_CONFIG=${xxx}
   ```

   其中"xxx"为配置文件的绝对路径，如：

   ```bash
   export MINDSPORE_DUMP_CONFIG=/path/to/data_dump.json
   ```

    注意：

    - 需要在执行benchmark之前，设置好环境变量，benchmark执行过程中设置将会不生效。

#### Dump数据目录结构

```text
{path}/
    - {net_name}/
        - {folder_id}/
            {op_name}_{input_output_index}_{shape}_{data_type}_{format}.bin
        ...
```

- `path`：`data_dump.json`配置文件中设置的绝对路径。
- `net_name`：`data_dump.json`配置文件中设置的网络名称。
- `folder_id`：默认创建编号为0的文件夹，每执行一次benchmark程序，该文件夹编号加1，以此类推，最多支持的文件夹数量为1000。
- `op_name`：算子名称。
- `input_output_index`：输入或输出标号，例如`output_0`表示该文件是该算子的第1个输出Tensor的数据。
- `data_type`：数据类型。
- `shape`：形状信息。
- `format`: 数据格式。

Dump生成的数据文件是后缀名为`.bin`的二进制文件，可以用Numpy的`np.fromfile()`接口读取数据，以数据类型为float32的bin文件为例：

```python
import numpy as np
np.fromfile("/path/to/dump.bin", np.float32)
```

## Windows环境使用说明

### 环境准备

使用Benchmark工具，需要进行如下环境准备工作。

- 编译：Benchmark工具代码在MindSpore源码的`mindspore/lite/tools/benchmark`目录中，参考构建文档中的[环境要求](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#环境要求-1)和[编译示例](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/build/build.html#编译示例-1)执行编译。
- 将推理需要的动态链接库加入环境变量PATH。

    ```bash
    set PATH=%PACKAGE_ROOT_PATH%\runtime\lib;%PATH%
    ```

    其中%PACKAGE_ROOT_PATH%是编译得到的包解压后的根目录。

### 参数说明

使用编译好的Benchmark工具进行模型的基准测试时，其命令格式如下所示。参数与Linux环境下使用一致，此处不再赘述。

```text
call benchmark.exe [--modelFile=<MODELFILE>] [--accuracyThreshold=<ACCURACYTHRESHOLD>]
   [--cosineDistanceThreshold=<COSINEDISTANCETHRESHOLD>]
   [--benchmarkDataFile=<BENCHMARKDATAFILE>] [--benchmarkDataType=<BENCHMARKDATATYPE>]
   [--cpuBindMode=<CPUBINDMODE>] [--device=<DEVICE>] [--help]
   [--inDataFile=<INDATAFILE>] [--loopCount=<LOOPCOUNT>]
   [--numThreads=<NUMTHREADS>] [--warmUpLoopCount=<WARMUPLOOPCOUNT>]
   [--enableFp16=<ENABLEFP16>] [--timeProfiling=<TIMEPROFILING>]
            [--inputShapes=<INPUTSHAPES>]
```

### 使用示例

对于不同的MindSpore Lite模型，在使用Benchmark工具对其进行基准测试时，可通过设置不同的参数，实现对其不同的测试功能。主要分为性能测试和精度测试，输出信息与Linux环境下一致，此处不再赘述。

#### 性能测试

- 使用随机输入，其他参数使用默认值。

    ```bat
    call benchmark.exe  --modelFile=/path/to/model.ms
    ```

- 使用随机输入，`timeProfiling`设为true，其他参数使用默认值。

    ```bat
    call benchmark.exe --modelFile=/path/to/model.ms --timeProfiling=true
    ```

#### 精度测试

输入数据通过`inDataFile`参数设定，标杆数据通过`benchmarkDataFile`参数设定。

- 指定了准确度阈值为3%。

    ```bat
    call benchmark.exe --modelFile=/path/to/model.ms --inDataFile=/path/to/input.bin --benchmarkDataFile=/path/to/output.out --accuracyThreshold=3
    ```

- 指定模型推理程序在CPU上运行。

    ```bat
    call benchmark.exe --modelFile=/path/to/model.ms --inDataFile=/path/to/input.bin --benchmarkDataFile=/path/to/output.out --device=CPU
    ```

- 指定输入数据的维度。

    ```bat
    call benchmark.exe --modelFile=/path/to/model.ms --inDataFile=/path/to/input.bin --benchmarkDataFile=/path/to/output.out --inputShapes=1,32,32,1
    ```

### Dump功能

Windows环境下Dump功能使用方法与[Linux环境](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_tool.html#dump功能)基本一致，此处不再赘述。

需注意的一点是，在Windows环境下，`data_dump.json`配置文件中设置绝对路径`Path`时，需指定为`\\`的形式。