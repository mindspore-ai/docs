# Error Analysis

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.5.0/docs/mindspore/source_en/model_train/debug/error_analysis/error_scenario_analysis.md)&nbsp;&nbsp;

As mentioned before, error analysis refers to analyzing and inferring possible error causes based on the obtained network and framework information (such as error messages and network code).

During error analysis, the first step is to identify the scenario where the error occurs and determine whether the error is caused by data loading and processing or network construction and training. You can determine whether it is a data or network problem based on the format of the error message. In the distributed parallel scenario, you can use a single-device execution network for verification. If there is no data loading and processing or network construction and training problem, the error is caused by a parallel scenario problem. The following describes the error analysis methods in different scenarios.

## Data Loading and Processing Error Analysis

When an error is reported during data processing, check whether C++ error messages are contained as shown in Figure 1. Typically, the name of the data processing operation using the C++ language is the same as that using Python. Therefore, you can determine the data processing operation that reports the error based on the error message and locate the error in the Python code.

![minddata-errmsg](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/docs/mindspore/source_zh_cn/model_train/debug/error_analysis/images/minddata_errmsg.png)

*Figure 1*

As shown in the following figure, `batch_op.cc` reports a C++ error. The batch operation combines multiple consecutive pieces of data in a dataset into a batch for data processing, which is implemented at the backend. According to the error description, the input data does not meet the parameter requirements of the batch operation. Data to be batch operated has the same shape, and the sizes of different shapes are displayed.

Data loading and processing has three phases: data preparation, data loading, and data augmentation. The following table lists common errors.

| Error Type| Error Description| Case Analysis|
|-------------|---------|---|
| Data preparation error| The dataset is faulty, involving a path or MindRecord file problem.| [Error Case](https://www.mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/minddata_debug.html)|
| Data loading error| Incorrect resource configuration, customized loading method, or iterator usage in the data loading phase.| [Error Case](https://www.mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/minddata_debug.html)|
| Data augmentation error| Unmatched data format/size, high resource usage, or multi-thread suspension.| [Error Case](https://www.mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/minddata_debug.html)|

## Network Construction and Training Error Analysis

The network construction and training process can be executed in the dynamic graph mode or static graph mode, and has two phases: build and execution. The error analysis method varies according to the execution phase in different modes.

The following table lists common network construction and training errors.

| Error Type  | Error Description| Case Analysis|
| - | - | - |
| Incorrect context configuration| An error occurs when the system configures the context.| [Error Analysis](https://mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/mindrt_debug.html)|
| Syntax error      | Python syntax errors and MindSpore static graph syntax errors, such as unsupported control flow syntax and tensor slicing errors| [Error Analysis](https://mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/mindrt_debug.html)|
| Operator build error  | The operator parameter value, type, or shape does not meet the requirements, or the operator function is restricted.| [Error Analysis](https://mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/mindrt_debug.html)|
| Operator execution error  | Input data exceptions, operator implementation errors, function restrictions, resource restrictions, etc.| [Error Analysis](https://mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/mindrt_debug.html)|
| Insufficient resources      | The device memory is insufficient, the number of function call stacks exceeds the threshold, and the number of flow resources exceeds the threshold.| [Error Analysis](https://mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/mindrt_debug.html)|

### Error Analysis of the Dynamic Graph Mode

In dynamic graph mode, the program is executed line by line according to the code writing sequence, and the execution result can be returned in time. Figure 2 shows the error message reported during dynamic graph build. The error message is from the Python frontend, indicating that the number of function parameters does not meet the requirements. Through the Python call stack, you can locate the error code: `c = self.mul(b, self.func(a,a,b))`.

Generally, the error message may contain `WARNING` logs. During error analysis, analyze the error message following Traceback first.

![pynative-errmsg](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/docs/mindspore/source_zh_cn/model_train/debug/error_analysis/images/pynative_errmsg.png)

*Figure 2*

In dynamic graph mode, common network construction and training errors are found in environment configuration, Python syntax, and operator usage. The general analysis method is as follows:

- Determine the object where the error is reported based on the error description, for example, the operator API name.
- Locate the code line where the error is reported based on the Python call stack information.
- Analyze the code input data and calculation logic at the position where the error occurs, and find the error cause based on the description and specifications of the error object in the [MindSpore API](https://www.mindspore.cn/docs/en/r2.5.0/api_python/mindspore.html).

### Error Analysis of the Static Graph Mode

In static graph mode, MindSpore builds the network structure into a computational graph, and then performs the computation operations involved in the graph. Therefore, errors reported in static graph mode include computational graph build errors and computational graph execution errors. Figure 3 shows the error message reported during computational graph build. When an error occurs, the `analyze_failed.ir` file is automatically saved to help analyze the location of the error code.

![graph-errmsg](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.5.0/docs/mindspore/source_zh_cn/model_train/debug/images/graph_errmsg.png)

*Figure 3*

The general error analysis method in static graph mode is as follows:

Check whether the error is caused by graph build or graph execution based on the error description.

- If the error is reported during computational graph build, analyze the cause and location of the failure based on the error description and the `analyze_failed.ir` file automatically saved when the error occurs.
- If the error is reported during computational graph execution, the error may be caused by insufficient resources or improper operator execution. You need to further distinguish the error based on the error message. If the error is reported during operator execution, locate the operator, use the dump function to save the input data of the operator, and analyze the cause of the error based on the input data.

For details about how to analyze and infer the failure cause, see the analysis methods described in [`analyze_failed.ir`](https://www.mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/mindir.html#how-to-derive-the-cause-of-the-failure-based-on-the-analyze-fail-ir-file-analysis-graph).

For details about how to use Dump to save the operator input data, see [Dump Function Debugging](https://www.mindspore.cn/docs/en/r2.5.0/model_train/debug/dump.html).

## Distributed Parallel Error Analysis

MindSpore provides the distributed parallel training function and supports multiple parallel modes. The following table lists common distributed parallel training errors and possible causes.

| Error Type| Error Description                        |
| ------------ | -------------------------------- |
| Incorrect policy configuration| Incorrect operator logic.|
|              | Incorrect scalar policy configuration.                |
|              | No policy configuration.        |
| Parallel script error| Incorrect script startup, or unmatched parallel configuration and startup task.|

### Incorrect Policy Configuration

Policy check errors may be reported after you enable automatic parallelism using `mindspore.set_auto_parallel_context(parallel_mode="semi_auto_parallel")`. These policy check errors are reported due to specific operator slicing restrictions. The following uses three examples to describe how to analyze the three types of errors.

#### Incorrect Operator Logic

The error message is as follows:

```python
[ERROR]Check StridedSliceInfo1414: When there is a mask, the input is not supported to be split
```

The following shows a piece of possible error code where the network input is a [2, 4] tensor. The network is sliced to obtain the first half of dimension 0 in the input tensor. It is equivalent to the x[:1, :]operation in NumPy, where x is the input tensor. On the network, the (2,1) policy is configured for the stridedslice operator to slice dimension 0.

```python
tensor = Tensor(ones((2, 4)))
stridedslice = ops.StridedSlice((0, 0),(1, 4), (1, 1))

class MyStridedSlice(nn.Cell):
    def __init__(self):
        super(MyStridedSlice, self).__init__()
        self.slice = stridedslice.shard(((2,1),))

    def construct(self, x):
        # x is a two-dimensional tensor
        return self.slice(x)
```

Error cause:

The piece of code performs the slice operation on dimension 0. However, the configured policy (2,1) indicates that the slice operation is performed on both dimension 0 and dimension 1 of the input tensor. According to the description of operator slicing in the [MindSpore API](https://www.mindspore.cn/docs/en/r2.5.0/api_python/operator_list_parallel.html),

> only the mask whose value is all 0s is supported. All dimensions that are sliced must be extracted together. The input dimensions whose strides is not set to 1 cannot be sliced.

Dimensions that are sliced cannot be separately extracted. Therefore, the policy must be modified as follows:

Change the policy of dimension 0 from 2 to 1. In this way, dimension 0 will be sliced into one, that is, dimension 0 will not be sliced. Therefore, the policy meets the operator restrictions and the policy check is successful.

```python
class MyStridedSlice(nn.Cell):
    def __init__(self):
        super(MyStridedSlice, self).__init__()
        self.slice = stridedslice.shard(((1,1),))

    def construct(self, x):
        # x is a two-dimensional tensor
        return self.slice(x)
```

#### Incorrect Scalar Policy Configuration

Error message:

```python
[ERROR] The strategy is ..., strategy len:. is not equal to inputs len:., index:
```

Possible error code:

```python
class MySub(nn.Cell):
    def __init__(self):
        super(MySub, self).__init__()
        self.sub = ops.Sub().shard(((1,1), (1,)))
    def construct(self, x):
        # x is a two-dimensional tensor
        return self.sub(x, 1)
```

The input of many operators can be scalars, such as addition, subtraction, multiplication, and division operations and axis of operators such as concat and gather. For such operations with scalar input, do not configure policies for these scalars. If the preceding method is used to configure a policy for the subtraction operation and the policy (1,) is configured for scalar 1, an error is reported.
That is, the length of the policy whose index is 1 is 1, which is not equal to the length 0 of the corresponding input. In this case, the input is a scalar.

Modified code:

In this case, set an empty policy for the scalar or do not set any policy (recommended method).

```python
self.sub = ops.Sub().shard(((1,1),()))

self.sub = ops.Sub().shard(((1,1),))
```

#### No Policy Configuration

```python
[ERROR]The strategy is ((8, 1)), shape 4 can not be divisible by strategy value 8
```

Possible error code:

```python
class MySub(nn.Cell):
    def __init__(self):
        super(MySub, self).__init__()
        self.sub = ops.Sub()
    def construct(self, x):
        # x is a two-dimensional tensor
        return self.sub(x, 1)
```

The following piece of code runs training in an 8-device environment in semi-automatic parallel mode. No policy is configured for the Sub operator in the example and the default policy of the Sub operator is data parallel. Assume that the input x is a matrix of size [2, 4]. After the build starts, an error is reported, indicating that the input dimensions are insufficient for slicing. In this case, you need to modify the policy as follows (ensure that the number of dimensions for slicing are less than that of the input tensor):

```python
class MySub(nn.Cell):
    def __init__(self):
        super(MySub, self).__init__()
        self.sub = ops.Sub().shard(((2, 1), ()))
    def construct(self, x):
        # x is a two-dimensional tensor
        return self.sub(x, 1)
```

(2, 1) indicates that dimension 0 of the first input tensor is sliced into two parts, and dimension 1 is sliced into one, that is, dimension 1 is not sliced. The second input of `ops.Sub` is a scalar that cannot be sliced. Therefore, the slicing policy is set to empty ().

### Parallel Script Error

The following is a piece of code for running an 8-device Ascend environment and using the bash script to start the training task.

```bash
#!/bin/bash
set -e
EXEC_PATH=$(pwd)
export RANK_SIZE=8
export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json

for((i=0;i<RANK_SIZE;i++))
do
    rm -rf device$i
    mkdir device$i
    cp ./train.py ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python ./train.py > train.log$i 2>&1 &
    cd ../
done
echo "The program launch succeed, the log is under device0/train.log0."
```

Errors may occur in the following scenarios:

1) The number of training tasks (`RANK_SIZE`) started using the for loop does not match the number of devices configured in the `rank_table_8pcs.json` configuration file. As a result, an error is reported.

2) The command for executing the training script is not executed in asynchronous mode (`python ./train.py > train.log$i 2>&1`). As a result, training tasks are started at different time, and an error is reported. In this case, add the `&` operator to the end of the command, indicating that the command is executed asynchronously in the subshell. In this way, multiple tasks can be started synchronously.

In parallel scenarios, you may encounter the `Distribute Task Failed` error. In this case, analyze whether the error occurs in the computational graph build phase or the execution phase of printing training loss to further locate the error.

For details, visit the following website:

For more information about distributed parallel errors in MindSpore, see [Distributed Task Failed](https://www.hiascend.com/developer/blog/details/0232107350747190117).

## CANN Error Analysis

> - This section is only applicable to Ascend platform, CANN (Compute Architecture for Neural Networks) is Huawei heterogeneous computing architecture for AI scenarios, and MindSpore of Ascend platform runs on top of CANN.

When running MindSpore on the Ascend platform, certain scenarios will encounter errors from the underlying CANN. Such errors are generally reported in the log with the `Ascend error occurred` keyword, and the error message consists of the error code and the error content, as follows:

```c++
[ERROR] PROFILER(138694,ffffaa6c8480,python):2022-01-10-14:19:56.741.053 [mindspore/ccsrc/profiler/device/ascend/ascend_profiling.cc:51] ReportErrorMessage] Ascend error occurred, error message:
EK0001: Path [/ms_test/csj/csj/user_scene/profiler_chinese_中文/resnet/scripts/train/data/profiler] for [profilerResultPath] is invalid or does not exist. The Path name can only contain A-Za-z0-9-_.
```

One of the CANN error codes consists of 6 characters, such as `EK0001` above, which contains three fields:

| Field 1 | Field 2 | Field 3 |
|:------|:------|:------ |
| Level (1 position) | Module (1 position) | Error code (4 positions) |

Among them, the level is divided into E, W, I, respectively, indicating error, alarm, prompt class. Module indicates the CANN module that reports errors, as shown in the following table:

| Err error code | CANN module | Err error code | CANN module |
|:------|:------|:------ |:------ |
| E10000-E19999 | GE | EE0000-EE9999 | runtime |
| E20000-E29999 | FE | EF0000-EF9999 | LxFusion |
| E30000-E39999 | AICPU | EG0000-EG9999 | mstune |
| E40000-E49999 | TEFusion | EH0000-EH9999 | ACL |
| E50000-E89999 | AICORE | EI0000-EJ9999 | HCCL&HCCP |
| E90000-EB9999 | TBE Compiling front and back ends | EK0000-EK9999 | Profiling |
| EC0000-EC9999 | Autotune | EL0000-EL9999 | Driver |
| ED0000-ED9999 | RLTune | EZ0000-EZ9999 | Operator public error |

> AICORE operator: The AI Core operator is the main component of the computational core of the Ascend AI processor and is responsible for performing computationally intensive operator related to vector and tensor.
> AICPU operator: AI CPU operator is the AI CPU responsible for executing CPU-like operator (including control operator, scalar and vector, and other general-purpose computations) in the Hayes SoC of the Ascend processor.

Among the 4-bit error codes, 0000~8999 are user-class errors and 9000~9999 are internal error codes. Generally, user-class error users can correct the error by themselves according to the error message, while internal error codes need to contact Huawei for troubleshooting. You can go to [MindSpore Community](https://gitee.com/mindspore) or [Ascend Community](https://gitee.com/ascend) to submit issue to get help. Some common error reporting scenarios are shown in the following table:

| Common Error Types   | Error Description | Case Analysis |
| - | - | - |
| AICORE Operator Compilation Problem | AICORE Operator Error During Compilation | [AICORE Operator Compilation Problem](https://www.mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/cann_error_cases.html#aicore-operator-compilation-problem)|
| AICORE Operator Execution Problem  | AICORE Operator Error During Execution| [AICORE Operator Execution Problem](https://mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/cann_error_cases.html#aicore-operator-execution-problem) |
| AICPU Operator Execution Problem   | AICPU Operator Error During Execution | [AICPU Operator Execution Problem](https://mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/cann_error_cases.html#aicpu-operator-execution-problem) |
| runtime FAQ   | Including input data exceptions, operator implementation errors, functional limitations, resource limitations, etc. | [runtime FAQ](https://mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/cann_error_cases.html#runtime-faq) |
| HCCL & HCCP FAQ   | Common communication problems during multi-machine multi-card training, including socket build timeout, notify wait timeout, ranktable configuration error, etc. | [HCCL & HCCP FAQ](https://mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/cann_error_cases.html#hccl-hccp-faq) |
| profiling FAQ    | Errors when running profiling for performance tuning | [profiling FAQ](https://mindspore.cn/docs/en/r2.5.0/model_train/debug/error_analysis/cann_error_cases.html#profiling-faq) |

For more information about CANN errors, refer to the [Ascend CANN Developer Documentation](https://www.hiascend.com/document/moreVersion/zh/CANNCommunityEdition/) to check the troubleshooting section of the corresponding CANN version.
