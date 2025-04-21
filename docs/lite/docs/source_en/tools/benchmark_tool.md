# benchmark

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/tools/benchmark_tool.md)

## Overview

After model conversion and before inference, you can use the Benchmark tool to perform benchmark testing on a MindSpore Lite model. It can not only perform quantitative analysis (performance) on the forward inference execution duration of a MindSpore Lite model, but also perform comparative error analysis (accuracy) based on the output of the specified model.

## Linux Environment Usage

### Environment Preparation

To use the Benchmark tool, you need to prepare the environment as follows:

- Compilation: Install build dependencies and perform build. The code of the Benchmark tool is stored in the `mindspore/lite/tools/benchmark` directory of the MindSpore source code. For details about the build operations, see the [Environment Requirements](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html#environment-requirements) and [Compilation Example](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html#compilation-example) in the build document.

- Run: Obtain the `benchmark` tool and configure environment variables. For details, see [Output Description](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html#environment-requirements) in the build document.

- Add the path of dynamic library required by the inference code to the environment variables LD_LIBRARY_PATH.

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/runtime/lib:${LD_LIBRARY_PATH}
    ````

    ${PACKAGE_ROOT_PATH} is the compiled inference package path after decompressing.

- If you're running this benchmark based on Ascend, use the following command to switch:

    ```bash
    export ASCEND_DEVICE_ID=$RANK_ID
    ```

### Parameter Description

The command used for benchmark testing based on the compiled Benchmark tool is as follows:

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

The following describes the parameters in detail.

| Parameter            | Attribute | Function                                                     | Parameter Type                                                 | Default Value | Value Range |
| ----------------- | ---- | ------------------------------------------------------------ | ------ | -------- | ------------------------------- |
| `--modelFile=<MODELFILE>` | Mandatory | Specifies the file path of the MindSpore Lite model for benchmark testing. | String | Null  | -        |
| `--accuracyThreshold=<ACCURACYTHRESHOLD>` | Optional | Specifies the accuracy threshold. | Float           | 0.5    | -        |
| `--cosineDistanceThreshold=<COSINEDISTANCETHRESHOLD>` | Optional | Specifies the cosine distance threshold, only when the parameter is specified and it's value is bigger than -1, the cosine distance is computed. | Float           | -1.1    | -        |
| `--benchmarkDataFile=<BENCHMARKDATAFILE>` | Optional | Specifies the file path of the benchmark data. The benchmark data, as the comparison output of the tested model, is output from the forward inference of the tested model under other deep learning frameworks using the same input. | String | Null | - |
| `--benchmarkDataType=<BENCHMARKDATATYPE>` | Optional | Specifies the calibration data type. | String | FLOAT | FLOAT, INT32, INT8 or UINT8|
| `--cpuBindMode=<CPUBINDMODE>` | Optional | Specifies the type of the CPU core bound to the model inference program. | Integer | 1      | 2: medium core<br/>1: large core<br/>0: not bound |
| `--device=<DEVICE>` | Optional | Specifies the type of the device on which the model inference program runs. | String | CPU | CPU or GPU or NPU or Ascend |
| `--help` | Optional | Displays the help information about the `benchmark` command. | - | - | - |
| `--inDataFile=<INDATAFILE>` | Optional | Specifies the file path of the input data of the tested model. If this parameter is not set, a random value will be used. | String | Null  | -       |
| `--loopCount=<LOOPCOUNT>` | Optional | Specifies the number of forward inference times of the tested model when the Benchmark tool is used for the benchmark testing. The value should be a positive integer. | Integer | 10 | - |
| `--numThreads=<NUMTHREADS>` | Optional | Specifies the number of threads for running the model inference program. | Integer | 2 | - |
| `--warmUpLoopCount=<WARMUPLOOPCOUNT>` | Optional | Specifies the number of preheating inference times of the tested model before multiple rounds of the benchmark test are executed. | Integer | 3 | - |
| `--enableFp16=<ENABLEFP16>` | Optional | Specifies whether the float16 operator is preferred. | Boolean | false | true, false |
| `--timeProfiling=<TIMEPROFILING>` | Optional | Specifies whether to use TimeProfiler to print every kernel's cost time. | Boolean | false | true, false |
| `--inputShapes=<INPUTSHAPES>` | Optional | Specifies the shape of input data, the format should be NHWC. Use "," to segregate each dimension of input shape, and for several input shapes, use ":" to segregate. | String | Null | - |
| `--perfProfiling=<PERFPROFILING>` | Optional | Specifies whether to use PerfProfiler to print every kernel's CPU performance data (PMU readings), it is disabled when timeProfiling is true. Only aarch64 CPU is supported. | Boolean | false | true, false |
| `--perfEvent=<PERFEVENT>` | Optional | Specifies what CPU performance data to measure when PerfProfiling is true. When set as CYCLE, the number of CPU cycles and instructions will be printed; when set as CACHE, cache reference times and cache miss times will be printed; when set as STALL, CPU front-end stall cycles and back-end stall cycles will be printed. | String | CYCLE | CYCLE/CACHE/STALL |
| `--decryptKey=<DECRYPTKEY>` | Optional | The key used to decrypt the model, in hexadecimal characters for the key. It only supports AES-GCM, and the key length is only 16Byte. | String | null | Note that the key is a string represented by hexadecimal. For example, if the key is defined as `b'0123456789ABCDEF'`, the corresponding hexadecimal representation is `30313233343536373839414243444546`. Users on the Linux platform can use the `xxd` tool to convert the key represented by the bytes to a hexadecimal representation. |
| `--cryptoLibPath=<CRYPTOLIBPATH>` | Optional | The path to the OpenSSL encryption library. | String | null | - |

### Example

When using the Benchmark tool to perform benchmark testing on different MindSpore Lite models, you can set different parameters to implement different test functions. The testing is classified into performance test and accuracy test.

#### Performance Test

The main test indicator of the performance test performed by the Benchmark tool is the duration of a single forward inference. In a performance test, you do not need to set benchmark data parameters such as `benchmarkDataFile`. But you can set the parameter `timeProfiling` as True or False to decide whether to print the running time of the model at the network layer on a certain device. The default value of `timeProfiling` is False. For example:

```bash
./benchmark --modelFile=/path/to/model.ms
```

This command uses a random input, and other parameters use default values. After this command is executed, the following statistics are displayed. The statistics include the minimum duration, maximum duration, and average duration of a single inference after the tested model runs for the specified number of inference rounds.

```text
Model = model.ms, numThreads = 2, MinRunTime = 72.228996 ms, MaxRuntime = 73.094002 ms, AvgRunTime = 72.556000 ms
```

```bash
./benchmark --modelFile=/path/to/model.ms --timeProfiling=true
```

This command uses a random input, sets the parameter `timeProfiling` as true, and other parameters use default values. After this command is executed, the statistics on the running time of the model at the network layer will be displayed as follows. In this case, the statistics are displayed by`opName` and `optype`. `opName` indicates the operator name, `optype` indicates the operator type, and `avg` indicates the average running time of the operator per single run, `percent` indicates the ratio of the operator running time to the total operator running time, `calledTimess` indicates the number of times that the operator is run, and `opTotalTime` indicates the total time that the operator is run for a specified number of times. Finally, `total time` and `kernel cost` show the average time consumed by a single inference operation of the model and the sum of the average time consumed by all operators in the model inference, respectively.

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

The accuracy test performed by the Benchmark tool focuses on setting [benchmark data (input.bin and output.out)](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/tools/benchmark_golden_data.html) to compare and verify the accuracy of the MindSpore Lite model output. In an accuracy test, in addition to the `modelFile` parameter, the `benchmarkDataFile` parameter must be set. For example:

```bash
./benchmark --modelFile=/path/to/model.ms --inDataFile=/path/to/input.bin --device=CPU --accuracyThreshold=3 --benchmarkDataFile=/path/to/output.out
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

To set specified input shapes (such as 1,32,32,1), use the command as follows:

```bash
./benchmark --modelFile=/path/to/model.ms --inDataFile=/path/to/input.bin --inputShapes=1,32,32,1 --device=CPU --accuracyThreshold=3 --benchmarkDataFile=/path/to/output.out
```

#### CPU Performance Test

The main test indicator of the CPU performance test performed by the Benchmark tool is the readings of CPU Performance Monitor Unit(PMU) of a single forward inference, including the number of CPU cycles and instructions, cache reference times and cache miss times, front-end stall cycles and back-end stall cycles. In a performance test, you do not need to set benchmark data parameters such as `benchmarkDataFile`. But you can set the parameter `perfProfiling` as True or False to decide whether to print the CPU performance data of the model at the network layer on a certain device, and set `perfEvent` as `CYCLE`/`CACHE`/`STALL` to decide what CPU performance data to measure. The default value of `perfProfiling` is False, the default value of `perfEvent` is `CYCLE`. Due to the fluctuation of PMU readings in multi-thread tests, `numThreads` is suggested to be set as `1`. For example:

```bash
./benchmark --modelFile=/path/to/model.ms --perfProfiling=true --numThreads=1
```

This command uses a random input, sets the parameter `perfProfiling` as true, and other parameters use default values. After this command is executed, the statistics on the running time of the model at the network layer will be displayed as follows. In this case, the statistics are displayed by`opName` and `optype`. `opName` indicates the operator name, `optype` indicates the operator type, and `cycles(k)` indicates the average CPU cycles of the operator per single run (in thousand, affected by CPU frequency), `cycles(%)` indicates the ratio of the operator CPU cycles to the total operator CPU cycles, `ins(k)` indicates the average CPU instructions of the operator per single run (in thousand), and `ins(%)` indicates the ratio of the operator CPU instructions to the total operator CPU instructions. Finally, `Model`/`NumThreads`/`MinRuntime`/`MaxRunTime`/`AvgRunTime` is presented for reference.

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

When `perfEvent` is set as `CACHE`, the columns will be `cache ref(k)`/`cache ref(%)`/`miss(k)`/`miss(%)`, which indicate cache reference times / cache reference ratio / cache miss times / cache miss ratio(to all cache misses, not to cache references); when `perfEvent` is set as `STALL`, the columns will be`frontend(k)`/`frontend(%)`/`backend(k)`/`backend(%)`, which indicate CPU front-end stall cycles / front-end stall cycles ratio / back-end stall cycles / back-end stall cycles ratio. For example:

```bash
./benchmark --modelFile=/path/to/model.ms --perfProfiling=true --perfEvent="CACHE"
```

```bash
./benchmark --modelFile=/path/to/model.ms --perfProfiling=true --perfEvent="STALL"
```

### Dump

Benchmark tool provides Dump function (currently only supports `CPU` and mobile `GPU` operators), which saves the input and output data of the operator in the model to a disk file. These files can be used to locate the problem of abnormal accuracy during the model inference process.

#### Dump Step

1. Create dump json file:`data_dump.json`, the name and location of the JSON file can be customized.

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

    - `dump_mode`: 0: dump all kernels in graph, 1: dump kernels in kernels list.
    - `path`: The absolute path to save dump data.
    - `net_name`: The net name, e.g.: ResNet50. If this field is not specified, the default value is "default".
    - `input_output`: 0: dump input and output of kernel, 1: dump input of kernel, 2: dump output of kernel.
    - `kernels`: List of operator names. If this field is not specified or the value is set to [], `"dump_mode"` must be set to 0; otherwise, the value of `"dump_mode"` must be set to 1.

2. Specify the json configuration file of Dump.

   ```bash
   export MINDSPORE_DUMP_CONFIG=${xxx}
   ```

   "xxx" represents the absolute path of data_dump.json, such as:

   ```bash
   export MINDSPORE_DUMP_CONFIG=/path/to/data_dump.json
   ```

    Note:

    - The environment variables need to be set before the benchmark is executed. The settings will not take effect during the execution of the benchmark.

#### Dump Data Directory Structure

```text
{path}/
    - {net_name}/
        - {folder_id}/
            {op_name}_{input_output_index}_{shape}_{data_type}_{format}.bin
        ...
```

- `path`: the absolute path set in the `data_dump.json` configuration file.
- `net_name`: the network name set in the `data_dump.json` configuration file.
- `folder_id`: The folder number created by default is 0. Each time the benchmark program is executed, the folder number is increased by 1, the maximum number of folders supported is 1000.
- `op_name`: the name of the operator.
- `input_output_index`: the index of input or output. For example, `output_0` means that the file is the data of the first output Tensor of the operator.
- `data_type`: the data type of the operator.
- `shape`: the shape of the operator.
- `format`: the format of the operator.

The data file generated by Dump is a binary file with the suffix `.bin`. You can use the `np.fromfile()` interface in Numpy to read the data. Take the bin file with the data type of float32 as an example:

```python
import numpy as np
np.fromfile("/path/to/dump.bin", np.float32)
```

## Windows Environment Usage

### Environment Preparation

To use the Benchmark tool, you need to prepare the environment as follows:

- Compilation: Install build dependencies and perform build. The code of the Benchmark tool is stored in the `mindspore/lite/tools/benchmark` directory of the MindSpore source code. For details about the build operations, see the [Environment Requirements](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html#environment-requirements) and [Compilation Example](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html#compilation-example) in the build document.
- Add the path of dynamic library required by the benchmark to the environment variables PATH.

    ````bash
    set PATH=%PACKAGE_ROOT_PATH%\runtime\lib;%PATH%
    ````

    %PACKAGE_ROOT_PATH% is the decompressed package path obtained by compiling.

### Parameter Description

The command used for benchmark testing based on the compiled Benchmark tool is as follows. The parameters are the same as those used in the Linux environment, and will not be repeated here.

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

### Example

When using the Benchmark tool to perform benchmark testing on different MindSpore Lite models, you can set different parameters to implement different test functions. The testing is classified into performance test and accuracy test. The output statistics are the same as those in the Linux environment, and will not be repeated here.

#### Performance Test

- Use a random input and default values for other parameters.

    ```bat
    call benchmark.exe --modelFile=/path/to/model.ms
    ```

- set `timeProfiling=true`, use a random input and default values for other parameters.

    ```bat
    call benchmark.exe --modelFile=/path/to/model.ms --timeProfiling=true
    ```

#### Accuracy Test

 The input data is set by the `inDataFile` parameter, and the calibration data is set by the `benchmarkDataFile` parameter.

- Set the accuracy threshold to 3%.

    ```bat
    call benchmark.exe --modelFile=/path/to/model.ms --inDataFile=/path/to/input.bin --benchmarkDataFile=/path/to/output.out --accuracyThreshold=3
    ```

- Run on the CPU.

    ```bat
    call benchmark.exe --modelFile=/path/to/model.ms --inDataFile=/path/to/input.bin --benchmarkDataFile=/path/to/output.out --device=CPU
    ```

- Set specified input shapes.

    ```bat
    call benchmark.exe --modelFile=/path/to/model.ms --inDataFile=/path/to/input.bin --benchmarkDataFile=/path/to/output.out --inputShapes=1,32,32,1
    ```

### Dump

The usage of Dump function in the Windows environment is basically the same as that of in the [Linux environment](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/tools/benchmark_tool.html#dump), and will not be repeated here.

Note that in the Windows environment, when setting the absolute path `Path` in the `data_dump.json` configuration file, it must be specified in the form of `\\`.