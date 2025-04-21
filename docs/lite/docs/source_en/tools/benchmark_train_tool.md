# benchmark_train

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/tools/benchmark_train_tool.md)

## Overview

The same as `benchmark`, you can use the `benchmark_train` tool to perform benchmark testing on a MindSpore ToD (Train on Device) model. It can not only perform quantitative analysis (performance) on the execution duration the model, but also perform comparative error analysis (accuracy) based on the output of the specified model.

## Linux Environment Usage

### Environment Preparation

To use the `benchmark_train` tool, you need to prepare the environment as follows:

- Compilation: The code of the `benchmark_train` tool is stored in the `mindspore/lite/tools/benchmark_train` directory of the MindSpore source code. For details about the build operations, see the [Environment Requirements](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html#environment-requirements) and [Compilation Example](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html#compilation-example) in the build document.

- Configure environment variables: For details, see [Output Description](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html#directory-structure-1) in the build document. Suppose the absolute path of MindSpore Lite training package you build is `/path/mindspore-lite-{version}-{os}-{arch}.tar.gz`, the commands to extract the package and configure the LD_LIBRARY_PATH variable are as follows:

    ```bash
    cd /path
    tar xvf mindspore-lite-{version}-{os}-{arch}.tar.gz
    export LD_LIBRARY_PATH=/path/mindspore-lite-{version}-{os}-{arch}/runtime/lib:/path/mindspore-lite-{version}-{os}-{arch}/runtime/third_party/libjpeg-turbo/lib:/path/mindspore-lite-{version}-{os}-{arch}/runtime/third_party/hiai_ddk/lib:/path/mindspore-lite-{version}-{os}-{arch}/runtime/third_party/glog:${LD_LIBRARY_PATH}
    ```

The absolute path of the benchmark_train tool is `/path/mindspore-lite-{version}-{os}-{arch}/tools/benchmark_train/benchmark_train`.

### Parameter Description

The command used for benchmark testing based on the compiled `benchmark_train` tool is as follows:

```text
./benchmark_train [--modelFile=<MODELFILE>] [--accuracyThreshold=<ACCURACYTHRESHOLD>]
   [--expectedDataFile=<BENCHMARKDATAFILE>] [--warmUpLoopCount=<WARMUPLOOPCOUNT>]
   [--timeProfiling=<TIMEPROFILING>] [--help]
   [--inDataFile=<INDATAFILE>] [--epochs=<EPOCHS>]
   [--exportFile=<EXPORTFILE>]
```

The following describes the parameters in detail.

| Parameter            | Attribute | Function                                                     | Parameter Type                                                 | Default Value | Value Range |
| ----------------- | ---- | ------------------------------------------------------------ | ------ | -------- | ---------------------------------- |
| `--modelFile=<MODELFILE>` | Mandatory | Specifies the file path of the MindSpore Lite model for benchmark testing. | String | Null  | -        |
| `--accuracyThreshold=<ACCURACYTHRESHOLD>` | Optional | Specifies the accuracy threshold. | Float           | 0.5    | -        |
| `--expectedDataFile=<BENCHMARKDATAFILE>` | Optional | Specifies the file path of the benchmark data. The benchmark data, as the comparison output of the tested model, is output from the forward inference of the tested model under other deep learning frameworks using the same input. | String | Null | - |
| `--help` | Optional | Displays the help information about the `benchmark_train` command. | - | - | - |
| `--warmUpLoopCount=<WARMUPLOOPCOUNT>` | Optional | Specifies the number of preheating inference times of the tested model before multiple rounds of the benchmark test are executed. | Integer | 3 | - |
| `--timeProfiling=<TIMEPROFILING>`  | Optional | Specifies whether to use TimeProfiler to print every kernel's cost time. | Boolean | false | true, false |
| `--inDataFile=<INDATAFILE>` | Optional | Specifies the file path of the input data of the tested model. If this parameter is not set, a random value will be used. | String | Null  | -       |
| `--epochs=<EPOCHS>` | Optional | Specifies the number of training epochs and print the consuming time. | Integer | 0 | >=0 |
| `--exportFile=<EXPORTFILE>` | Optional | Specifies the path of exporting file. | String | Null | - |

### Example

When using the `benchmark_train` tool to perform benchmark testing, you can set different parameters to implement different test functions. The testing is classified into performance test and accuracy test.

#### Performance Test

The main test indicator of the performance test performed by the Benchmark tool is the duration of a single forward inference. In a performance test, you do not need to set benchmark data parameters such as `benchmarkDataFile`. But you can set the parameter `timeProfiling` as True or False to decide whether to print the running time of the model at the network layer on a certain device. The default value of `timeProfiling` is False. For example:

```bash
./benchmark_train --modelFile=./models/test_benchmark.ms --epochs=10
```

This command uses a random input, and other parameters use default values. After this command is executed, the following statistics are displayed. The statistics include the minimum duration, maximum duration, and average duration of a single inference after the tested model runs for the specified number of inference rounds.

```text
Model = test_benchmark.ms, numThreads = 1, MinRunTime = 72.228996 ms, MaxRuntime = 73.094002 ms, AvgRunTime = 72.556000 ms
```

```bash
./benchmark_train --modelFile=./models/test_benchmark.ms --epochs=10 --timeProfiling=true
```

This command uses a random input, sets the parameter `timeProfiling` as true,  times and other parameters use default values. After this command is executed, the statistics on the running time of the model at the network layer will be displayed as follows. In this case, the statistics are displayed by`opName` and `optype`. `opName` indicates the operator name, `optype` indicates the operator type, and `avg` indicates the average running time of the operator per single run, `percent` indicates the ratio of the operator running time to the total operator running time, `calledTimess` indicates the number of times that the operator is run, and `opTotalTime` indicates the total time that the operator is run for a specified number of times. Finally, `total time` and `kernel cost` show the average time consumed by a single inference operation of the model and the sum of the average time consumed by all operators in the model inference, respectively.

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

#### Accuracy Test

The accuracy test performed by the Benchmark tool aims to verify the accuracy of the MindSpore model output by setting benchmark data (the default input and benchmark data type are float32). In an accuracy test, in addition to the `modelFile` parameter, the `inDataFile` and `expectedDataFile` parameters must be set. For example:

```bash
./benchmark_train --modelFile=./models/test_benchmark.ms --inDataFile=./input/test_benchmark.bin --accuracyThreshold=3 --expectedDataFile=./output/test_benchmark.out
```

This command specifies the input data and benchmark data of the tested model, specifies that the model inference program runs on the CPU, and sets the accuracy threshold to 3%. After this command is executed, the following statistics are displayed, including the single input data of the tested model, output result and average deviation rate of the output node, and average deviation rate of all nodes.

```text
InData0: 139.947 182.373 153.705 138.945 108.032 164.703 111.585 227.402 245.734 97.7776 201.89 134.868 144.851 236.027 18.1142 22.218 5.15569 212.318 198.43 221.853
================ Comparing Output data ================
Data of node age_out : 5.94584e-08 6.3317e-08 1.94726e-07 1.91809e-07 8.39805e-08 7.66035e-08 1.69285e-07 1.46246e-07 6.03796e-07 1.77631e-07 1.54343e-07 2.04623e-07 8.89609e-07 3.63487e-06 4.86876e-06 1.23939e-05 3.09981e-05 3.37098e-05 0.000107102 0.000213932 0.000533579 0.00062465 0.00296401 0.00993984 0.038227 0.0695085 0.162854 0.123199 0.24272 0.135048 0.169159 0.0221256 0.013892 0.00502971 0.00134921 0.00135701 0.000383242 0.000163475 0.000136294 9.77864e-05 8.00793e-05 5.73874e-05 3.53858e-05 2.18535e-05 2.04467e-05 1.85286e-05 1.05075e-05 9.34751e-06 6.12732e-06 4.55476e-06
Mean bias of node age_out : 0%
Mean bias of all nodes: 0%
=======================================================
```

### Dump

For specific usage, please refer to [Dump](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/tools/benchmark_tool.html#dump) of `benchmark` tool.
