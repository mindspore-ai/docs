# benchmark

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/mindir/benchmark_tool.md)

## Overview

Before performing inference after converting the model, you can use the Benchmark tool to benchmark the MindSpore Lite cloud-side inference model. It allows not only quantitative analysis of the forward inference execution time of MindSpore Lite cloud-side inference models (performance), but also comparable error analysis (accuracy) by specifying the model output.

## Linux Environment Usage Instructions

### Environment Preparation

To use the Benchmark tool, you need to do the following environment preparation work.

- Compile: The Benchmark tool code is in the `mindspore/lite/tools/benchmark` directory of the MindSpore source code. Refer to the build documentation for [Environment requirements](https://www.mindspore.cn/lite/docs/en/master/mindir/build.html#environment-requirements) and [Compilation Examples](https://www.mindspore.cn/lite/docs/en/master/mindir/build.html#compilation-examples) in the build documentation to perform the compilation.

- Run: Refer to [compilation output](https://www.mindspore.cn/lite/docs/en/master/mindir/build.html#directory-structure) in the build documentation to get the `benchmark` tool from the compiled package.

- Add the dynamic link libraries needed for inference to the environment variable LD_LIBRARY_PATH.

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/runtime/lib:${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ```

    ${PACKAGE_ROOT_PATH} is the root directory of the compiled package after unpacking.

- If benchmarking based on Ascend, use the following command to switch:

    ```bash
    export ASCEND_DEVICE_ID=$RANKK_ID
    ```

### Description of Parameters

When using the compiled Benchmark tool to benchmark the model, the command format is shown below.

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

Detailed parameter descriptions are provided below.

| Parameter names            | Properties | Function Descriptions                                                     | Tpyes of parameters                                                 | Default values | Value range |  Remarks  |
| ----------------- | ---- | ------------------------------------------------------------ | ------ | -------- | --------------------- | ---- |
| `--modelFile=<MODELPATH>` | Required | Specify the path to the MindSpore Lite model file that needs to be benchmarked. | String | null  | -        |
| `--accuracyThreshold=<ACCURACYTHRESHOLD>` | Optional | Specify the accuracy threshold. | Float           | 0.5    | -        |
| `--benchmarkDataFile=<CALIBDATAPATH>` | Optional | Specify the file path to the benchmark data. The benchmark data is used as the comparison output for this test model, which is derived from the same input and forward inference from other deep learning frameworks. | String | null | - |
| `--benchmarkDataType=<CALIBDATATYPE>` | Optional | Specify the benchmark data type. | String | FLOAT | FLOAT, INT32, INT8, UINT8 |
| `--device=<DEVICE>` | Optional | Specify the type of device on which the model inference program runs. | String | CPU | CPU, GPU, NPU, Ascend |
| `--help` | Optional | Display help information for the `benchmark` command. | - | - | - |
| `--inDataFile=<INDATAPATH>` | Optional | Specify the file path to the test model input data. If not set, random input is used. | String | null | - |
| `--loopCount=<LOOPCOUNT>` | Optional | Specify the number of forward inference runs for the test model when Benchmark tool performs benchmarking, with a positive integer value. | Integer | 10 | - |
| `--numThreads=<NUMTHREADS>` | Optional | Specify the number of threads on which the model inference program will run. | Integer | 2 | - |
| `--warmUpLoopCount=<WARMUPLOOPCOUNT>` | Optional | Specify the number of model warm-up inferences performed by the test model before executing the number of rounds of the benchmark test run. | Integer | 3 | - |
| `--inputShapes=<INPUTSHAPES>` | Optional | Specify the input dimensions, which should follow the original model format. The dimension values are separated by ',' and multiple input dimensions are separated by ':' | String | Null | - |
| `--cosineDistanceThreshold=<COSINEDISTANCETHRESHOLD>` | Optional | Specify the cosine distance threshold. The cosine distance will be calculated only if this parameter is specified and its value is greater than -1. | Float           | -1.1    | -        | Not supported at the moment |
| `--cpuBindMode=<CPUBINDMODE>` | Optional | Specify the type of CPU core to which the model inference program is bound when it runs. | Integer | 1      | 2: means medium core<br/>1: means large core<br/>0: means no binding | Not supported at the moment |
| `--enableFp16=<FP16PIORITY>` | Optional | Specify whether to give preference to float16 operator. | Boolean | false | true, false | Not supported at the moment |
| `--timeProfiling=<TIMEPROFILING>`  | Optional | Take effect at performance verification, specifying whether to use TimeProfiler to print the time of each operator. | Boolean | false | true, false | Not supported at the moment |
| `--perfEvent=<PERFEVENT>` | Optional | Take effect during CPU performance verification. Specify the specific content of the CPU performance parameters printed by PerfProfiler. When specified as CYCLE, the number of CPU cycles and the number of instructions of the operator will be printed. When specified as CACHE, the number of cache reads and the number of cache misses of the operator will be printed. When specified as STALL, the number of CPU front-end waiting cycles and the number of back-end waiting cycles will be printed. | String | CYCLE | CYCLE/CACHE/STALL | Not supported at the moment |
| `--decryptKey=<DECRYPTKEY>` | Optional | The key used to decrypt the file, expressed in hexadecimal characters. Only AES-GCM is supported, and the key length is only 16Byte. | String | null | Note that the key is a hexadecimal representation of the string, such as the key is defined as `b'0123456789ABCDEF'` corresponding to the hexadecimal representation of `30313233343536373839414243444546`. Linux platform users can use the `xxd` tool to convert the byte representation of the key to hexadecimal expression. | Not supported at the moment |
| `--cryptoLibPath=<CRYPTOLIBPATH>` | Optional | Path to the OpenSSL crypto library crypto | String | null | - | Not supported at the moment |

### Usage Examples

For different MindSpore Lite cloud-side inference models, when benchmarking them with Benchmark tool, different parameters can be set to achieve different testing functions for them, mainly divided into performance test and accuracy test.

#### Performance Test

The main test metric of the performance test performed by the Benchmark tool is the time taken by the model for a single forward inference. In the performance test task, there is no need to set benchmark data parameters such as `benchmarkDataFile`. For example:

```bash
./benchmark --modelFile=/path/to/model.mindir
```

This command uses random input and uses default values for other parameters. This command outputs the following statistics after execution, which shows the shortest single inference time, the longest single inference time and the average inference time of the test model after running the specified number of inference rounds.

```text
Model = model.mindir, numThreads = 2, MinRunTime = 72.228996 ms, MaxRuntime = 73.094002 ms, AvgRunTime = 72.556000 ms
```

#### Accuracy Test

The accuracy test performed by the Benchmark tool is mainly to compare and verify the accuracy of MindSpore Lite cloud-side inference model output by setting benchmark data. In the accuracy test task, in addition to the `modelFile` parameter, the `benchmarkDataFile` parameter must also be set. For example:

```bash
./benchmark --modelFile=/path/to/model.mindir --inDataFile=/path/to/input.bin --device=CPU --accuracyThreshold=3 --benchmarkDataFile=/path/to/output.out
```

This command specifies the input data and benchmark data of the test model, (the default input and benchmark data types are float32), and also specifies the model inference program to run on the CPU with an accuracy threshold of 3%. The command outputs the following statistics after execution, which shows the single input data of the test model, the output results and average deviation rate of the output nodes, and the average deviation rate of all nodes.

```text
InData0: 139.947 182.373 153.705 138.945 108.032 164.703 111.585 227.402 245.734 97.7776 201.89 134.868 144.851 236.027 18.1142 22.218 5.15569 212.318 198.43 221.853
================ Comparing Output data ================
Data of node age_out : 5.94584e-08 6.3317e-08 1.94726e-07 1.91809e-07 8.39805e-08 7.66035e-08 1.69285e-07 1.46246e-07 6.03796e-07 1.77631e-07 1.54343e-07 2.04623e-07 8.89609e-07 3.63487e-06 4.86876e-06 1.23939e-05 3.09981e-05 3.37098e-05 0.000107102 0.000213932 0.000533579 0.00062465 0.00296401 0.00993984 0.038227 0.0695085 0.162854 0.123199 0.24272 0.135048 0.169159 0.0221256 0.013892 0.00502971 0.00134921 0.00135701 0.000383242 0.000163475 0.000136294 9.77864e-05 8.00793e-05 5.73874e-05 3.53858e-05 2.18535e-05 2.04467e-05 1.85286e-05 1.05075e-05 9.34751e-06 6.12732e-06 4.55476e-06
Mean bias of node age_out : 0%
Mean bias of all nodes: 0%
=======================================================
```

If you need to specify the dimension of the input data (e.g. input dimension is 1, 32, 32, 1), use the following command:

```bash
./benchmark --modelFile=/path/to/model.mindir --inDataFile=/path/to/input.bin --inputShapes=1,32,32,1 --device=CPU --accuracyThreshold=3 --benchmarkDataFile=/path/to/output.out
```

If the model is encryption model, inference is performed after both `decryptKey` and `cryptoLibPath` are configured to decrypt the model. For example:

```bash
./benchmark --modelFile=/path/to/encry_model.mindir --decryptKey=30313233343536373839414243444546 --cryptoLibPath=/root/anaconda3/bin/openssl
```