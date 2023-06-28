# Performing Benchmark Testing

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/lite/source_en/use/benchmark_tool.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

After model conversion and before inference, you can use the Benchmark tool to perform benchmark testing on a MindSpore Lite model. It can not only perform quantitative analysis (performance) on the forward inference execution duration of a MindSpore Lite model, but also perform comparative error analysis (accuracy) based on the output of the specified model.

## Environment Preparation

To use the Benchmark tool, you need to prepare the environment as follows:

- Compilation: Install build dependencies and perform build. The code of the Benchmark tool is stored in the `mindspore/lite/tools/benchmark` directory of the MindSpore source code. For details about the build operations, see the [Environment Requirements](https://www.mindspore.cn/tutorial/lite/en/r1.0/use/build.html#environment-requirements) and [Compilation Example](https://www.mindspore.cn/tutorial/lite/en/r1.0/use/build.html#compilation-example) in the build document.

- Run: Obtain the `Benchmark` tool and configure environment variables. For details, see [Output Description](https://www.mindspore.cn/tutorial/lite/en/r1.0/use/build.html#output-description) in the build document.

## Example

When using the Benchmark tool to perform benchmark testing on different MindSpore Lite models, you can set different parameters to implement different test functions. The testing is classified into performance test and accuracy test.

### Performance Test

The main test indicator of the performance test performed by the Benchmark tool is the duration of a single forward inference. In a performance test, you do not need to set benchmark data parameters such as `calibDataPath`. For example:

```bash
./benchmark --modelPath=./models/test_benchmark.ms
```

This command uses a random input, and other parameters use default values. After this command is executed, the following statistics are displayed. The statistics include the minimum duration, maximum duration, and average duration of a single inference after the tested model runs for the specified number of inference rounds.

```
Model = test_benchmark.ms, numThreads = 2, MinRunTime = 72.228996 ms, MaxRuntime = 73.094002 ms, AvgRunTime = 72.556000 ms
```

### Accuracy Test

The accuracy test performed by the Benchmark tool is to verify the accuracy of the MinSpore model output by setting benchmark data (the default input and benchmark data type are float32). In an accuracy test, in addition to the `modelPath` parameter, the `calibDataPath` parameter must be set. For example:

```bash
./benchmark --modelPath=./models/test_benchmark.ms --inDataPath=./input/test_benchmark.bin --device=CPU --accuracyThreshold=3 --calibDataPath=./output/test_benchmark.out
```

This command specifies the input data and benchmark data of the tested model, specifies that the model inference program runs on the CPU, and sets the accuracy threshold to 3%. After this command is executed, the following statistics are displayed, including the single input data of the tested model, output result and average deviation rate of the output node, and average deviation rate of all nodes.

```
InData0: 139.947 182.373 153.705 138.945 108.032 164.703 111.585 227.402 245.734 97.7776 201.89 134.868 144.851 236.027 18.1142 22.218 5.15569 212.318 198.43 221.853
================ Comparing Output data ================
Data of node age_out : 5.94584e-08 6.3317e-08 1.94726e-07 1.91809e-07 8.39805e-08 7.66035e-08 1.69285e-07 1.46246e-07 6.03796e-07 1.77631e-07 1.54343e-07 2.04623e-07 8.89609e-07 3.63487e-06 4.86876e-06 1.23939e-05 3.09981e-05 3.37098e-05 0.000107102 0.000213932 0.000533579 0.00062465 0.00296401 0.00993984 0.038227 0.0695085 0.162854 0.123199 0.24272 0.135048 0.169159 0.0221256 0.013892 0.00502971 0.00134921 0.00135701 0.000383242 0.000163475 0.000136294 9.77864e-05 8.00793e-05 5.73874e-05 3.53858e-05 2.18535e-05 2.04467e-05 1.85286e-05 1.05075e-05 9.34751e-06 6.12732e-06 4.55476e-06
Mean bias of node age_out : 0%
Mean bias of all nodes: 0%
=======================================================
```


## Parameter Description

The command used for benchmark testing based on the compiled Benchmark tool is as follows:

```bash
./benchmark [--modelPath=<MODELPATH>] [--accuracyThreshold=<ACCURACYTHRESHOLD>]
			[--calibDataPath=<CALIBDATAPATH>] [--calibDataType=<CALIBDATATYPE>]
			[--cpuBindMode=<CPUBINDMODE>] [--device=<DEVICE>] [--help]
			[--inDataPath=<INDATAPATH>] [--loopCount=<LOOPCOUNT>]
			[--numThreads=<NUMTHREADS>] [--warmUpLoopCount=<WARMUPLOOPCOUNT>]
			[--fp16Priority=<FP16PRIORITY>]
```

The following describes the parameters in detail.

| Parameter            | Attribute | Function                                                     | Parameter Type                                                 | Default Value | Value Range |
| ----------------- | ---- | ------------------------------------------------------------ | ------ | -------- | ---------------------------------- |
| `--modelPath=<MODELPATH>` | Mandatory | Specifies the file path of the MindSpore Lite model for benchmark testing. | String | Null  | -        |
| `--accuracyThreshold=<ACCURACYTHRESHOLD>` | Optional | Specifies the accuracy threshold. | Float           | 0.5    | -        |
| `--calibDataPath=<CALIBDATAPATH>` | Optional | Specifies the file path of the benchmark data. The benchmark data, as the comparison output of the tested model, is output from the forward inference of the tested model under other deep learning frameworks using the same input. | String | Null | - |
| `--calibDataType=<CALIBDATATYPE>` | Optional | Specifies the calibration data type. | String | FLOAT | UINT8, FLOAT or INT8 |
| `--cpuBindMode=<CPUBINDMODE>` | Optional | Specifies the type of the CPU core bound to the model inference program. | Integer | 1      | −1: medium core<br/>1: large core<br/>0: not bound |
| `--device=<DEVICE>` | Optional | Specifies the type of the device on which the model inference program runs. | String | CPU | CPU or GPU |
| `--help` | Optional | Displays the help information about the `benchmark` command. | - | - | - |
| `--inDataPath=<INDATAPATH>` | Optional | Specifies the file path of the input data of the tested model. If this parameter is not set, a random value will be used. | String | Null  | -       |
| `--loopCount=<LOOPCOUNT>` | Optional | Specifies the number of forward inference times of the tested model when the Benchmark tool is used for the benchmark testing. The value is a positive integer. | Integer | 10 | - |
| `--numThreads=<NUMTHREADS>` | Optional | Specifies the number of threads for running the model inference program. | Integer | 2 | - |
| `--warmUpLoopCount=<WARMUPLOOPCOUNT>` | Optional | Specifies the number of preheating inference times of the tested model before multiple rounds of the benchmark test are executed. | Integer | 3 | - |
| `--fp16Priority=<FP16PIORITY>` | Optional | Specifies whether the float16 operator is preferred. | Bool | false | true, false |