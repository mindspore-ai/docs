# TimeProfiler Tool

<a href="https://gitee.com/mindspore/docs/blob/r0.7/lite/tutorials/source_en/use/timeprofiler_tool.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

The TimeProfiler tool can be used to analyze the time consumption of forward inference at the network layer of a MindSpore Lite model. The analysis is implemented using the C++ language.

## Environment Preparation

To use the TimeProfiler tool, you need to prepare the environment as follows:

- Compilation: Install build dependencies and perform build. The code of the TimeProfiler tool is stored in the `mindspore/lite/tools/time_profile` directory of the MindSpore source code. For details about the build operations, see the [Environment Requirements](https://www.mindspore.cn/lite/tutorial/en/r0.7/build.html#environment-requirements) and [Compilation Example](https://www.mindspore.cn/lite/tutorial/en/r0.7/build.html#compilation-example) in the build document.

- Run: Obtain the `timeprofile` tool and configure environment variables by referring to [Output Description](https://www.mindspore.cn/lite/tutorial/en/r0.7/build.html#output-description) in the build document.

## Parameter Description

The command used for analyzing the time consumption of forward inference at the network layer based on the compiled TimeProfiler tool is as follows:

```bash
./timeprofile --modelPath=<MODELPATH> [--help] [--loopCount=<LOOPCOUNT>] [--numThreads=<NUMTHREADS>] [--cpuBindMode=<CPUBINDMODE>] [--inDataPath=<INDATAPATH>] [--fp16Priority=<FP16PRIORITY>]
```

The following describes the parameters in detail.

| Parameter            | Attribute | Function                                                     | Parameter Type | Default Value | Value Range |
| ----------------- | ---- | ------------------------------------------------------------ | ------ | -------- | ---------------------------------- |
| `--help` | Optional | Displays the help information about the `timeprofiler` command. | - | - | - |
| `--modelPath=<MODELPATH> ` | Mandatory | Specifies the file path of the MindSpore Lite model for time consumption analysis. | String | Null  | -        |
| `--loopCount=<LOOPCOUNT>` | Optional | Specifies the number of times that model inference is executed when the TimeProfiler tool is used for time consumption analysis. The value is a positive integer. | Integer | 100 | - |
| `--numThreads=<NUMTHREADS>` | Optional | Specifies the number of threads for running the model inference program. | Integer | 4 | - |
| `--cpuBindMode=<CPUBINDMODE>` | Optional | Specifies the type of the CPU core bound to the model inference program. | Integer | 1      | −1: medium core<br/>1: large core<br/>0: not bound |
| `--inDataPath=<INDATAPATH>` | Optional | Specifies the file path of the input data of the specified model. If this parameter is not set, a random value will be used. | String | Null  | -        |
| `--fp16Priority=<FP16PRIORITY>` | Optional | Specifies whether the float16 operator is preferred. | Bool | false | true, false |

## Example

Take the `test_timeprofiler.ms` model as an example and set the number of model inference cycles to 10. The command for using TimeProfiler to analyze the time consumption at the network layer is as follows:

```bash
./timeprofile --modelPath=./models/test_timeprofiler.ms --loopCount=10
```

After this command is executed, the TimeProfiler tool outputs the statistics on the running time of the model at the network layer. In this example, the command output is as follows: The statistics are displayed by`opName` and `optype`. `opName` indicates the operator name, `optype` indicates the operator type, and `avg` indicates the average running time of the operator per single run, `percent` indicates the ratio of the operator running time to the total operator running time, `calledTimess` indicates the number of times that the operator is run, and `opTotalTime` indicates the total time that the operator is run for a specified number of times. Finally, `total time` and `kernel cost` show the average time consumed by a single inference operation of the model and the sum of the average time consumed by all operators in the model inference, respectively.

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