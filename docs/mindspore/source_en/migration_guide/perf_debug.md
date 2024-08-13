# Performance Tuning

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/perf_debug.md)

## FAQs and Solutions

- The following common problems may be encountered during the performance debugging phase:
    - The first step is time-consuming
         This phase mainly completes operations such as graph conversion, graph fusion, graph optimization, etc, which is the process of generating executable models. Refer to [How to Optimize Compilation Performance](https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph_syntax/static_graph_expert_programming.html#how-to-optimize-compilation-performance).
    - Iteration gap is time-consuming
         Most of the time consumption in this phase comes from data acquisition, see [Data Processing Performance Optimization](https://www.mindspore.cn/docs/en/master/model_train/dataset/optimize.html).
    - Forward and reverse computation is time-consuming
         This phase mainly executes the forward and reverse operators in the network and carries the main computational work of an iteration. Information such as operator time consumption during training can be recorded to a file via [Profiler](https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling.html). The performance data provides the performance data of the framework host execution and operator execution, which can also be viewed and analyzed by users through the [MindInsight](https://www.mindspore.cn/mindinsight/docs/en/master/index.html) visualization interface, helping users to debug neural network performance more efficiently.
    - Iteration trailing is time-consuming
         This phase is time consuming, which may be caused by the collection communication, and you can set the fusion policy to optimize. Refer to [all_reduce_fusion_config set allreduce fusion policy](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_auto_parallel_context.html).

## Performance Tuning Process

Firstly, it is necessary to obtain performance data, as shown in the specific acquisition method: [Performance Profiling (Ascend)](https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_ascend.html) and [Performance Profiling (GPU)](https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_gpu.html).

The performance tuning directions are as follows:

1. Operator performance tuning
2. Framework enabling performance tuning
3. Multi-Node synchronization performance tuning
4. Data processing performance tuning

For details, see [ResNet Network Migration](https://www.mindspore.cn/docs/en/master/migration_guide/sample_code.html).

> Some networks are large. In this case, the build is slow in graph mode. During performance tuning, distinguish graph build from network execution. This section describes the performance tuning policies in the network execution phase.

### Operator Performance Tuning

If a single operator takes a long time and the performance of the same operator varies greatly in different shapes or data types, the problem is caused by the operator performance. The solution is as follows:

1. Use data types with less computational workload. For example, if there is no obvious difference between the precision of the same operator in float16 and float32 modes, you can use the float16 format with less calculation workload.
2. Use other operators with the same algorithm to avoid this problem.
3. Pay attention to 16-alignment in the Ascend environment. Due to the design of the Ascend AI Processors, it is recommended that the calculation on the AI core be 16-alignment (each dimension in the shape is a multiple of 16).

If you find an operator with poor performance, you are advised to contact [MindSpore community](https://gitee.com/mindspore/mindspore/issues) for feedback. We will optimize the operator in time after confirming that the problem is caused by poor performance.

### Framework Enabling Performance Tuning

- Using the Static Graph Mode

  Generally, MindSpore in static graph mode is much faster than that in PyNative mode. It is recommended that training and inference be performed in static graph mode. For details, see [Combination of Dynamic and Static Graphs](https://mindspore.cn/docs/en/master/design/dynamic_graph_and_static_graph.html).

- On-device Execution

  MindSpore provides an [on-device execution method](https://www.mindspore.cn/docs/en/master/design/overview.html) to concurrently process data and execute the network on the device. You only need to set `dataset_sink_mode=True` in `model.train`. Note that this configuration is `False` by default. When this configuration is enabled, one epoch returns the result of only one network. You are advised to change the value to `False` during debugging.

- Using Automatic Mixed Precision

  The mixed precision training method accelerates the deep neural network training process by mixing the single-precision floating-point data format and the half-precision floating-point data format without compromising the network accuracy. Mixed precision training can accelerate the computing process, reduce memory usage and retrieval, and enable a larger model or batch size to be trained on specific hardware.

  For details, see [Mixed Precision Tutorial](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html).

- Enabling Graph Kernel Fusion

  Graph kernel fusion is a unique network performance optimization technology of MindSpore. It can automatically analyze and optimize the logic of existing network computational graphs, simplify and replace computational graphs, split and fuse operators, and build operators in a special way based on the target hardware capability to improve the computing resource utilization of devices and optimize the overall network performance. Compared with traditional optimization technologies, the graph kernel fusion technology has unique advantages, such as joint optimization of multiple operators across boundaries, cross-layer collaboration with operator compilation, and real-time compilation of operators based on Polyhedral. In addition, the entire optimization process of graph kernel fusion can be automatically completed after users enable the corresponding configuration. Network developers do not need to perform extra perception, so that users can focus on network algorithm implementation.

  Graph kernel fusion applies to scenarios that have high requirements on network execution time. Basic operators are combined to implement customized combination operators and these basic operators are automatically fused to improve the performance of the customized combination operators.

  For details, see [Graph Kernel Fusion Tutorial](https://mindspore.cn/docs/en/master/design/graph_fusion_engine.html).

- Others

  If there are too many conversion operators (TransData and Cast operators) and the conversion takes a long time, analyze the necessity of the manually added Cast operator. If the accuracy is not affected, delete the redundant Cast and TransData operators.

  If there are too many conversion operators automatically generated by MindSpore, the MindSpore framework may not be fully optimized for some special cases. In this case, contact [MindSpore community](https://gitee.com/mindspore/mindspore/issues) for feedback.

  In dynamic shape scenario, continuous graph build is required, which may cause a long end-to-end training time. You are advised to [avoid dynamic shape](https://www.mindspore.cn/docs/en/master/migration_guide/dynamic_shape.html).

### Multi-Node Synchronization Performance Tuning

During distributed training, after forward propagation and gradient calculation are complete in a step training process, each machine starts to perform AllReduce gradient synchronization. The AllReduce synchronization time is mainly affected by the number of weights and machines. For a more complex network with a larger machine scale, the AllReduce gradient update time is longer. In this case, you can perform AllReduce segmentation to reduce the time consumption.

In normal cases, AllReduce gradient synchronization waits until all backward operators are executed. That is, after the gradient of all gradients is calculated, the gradients of all machines are synchronized at a time. After AllReduce segmentation is used, the gradients of some weights can be calculated, gradient synchronization of this part of weights is immediately performed. In this way, gradient synchronization and gradient calculation of remaining operators can be performed concurrently, and this part of AllReduce gradient synchronization time is hidden. The shard strategy is usually manually tried to find an optimal solution (more than two shards are supported).
The [ResNet-50](https://gitee.com/mindspore/models/blob/master/official/cv/ResNet/train.py) is used as an example. The network has 160 weights. [85, 160] indicates that gradient synchronization is performed immediately after the gradients of weights 0 to 85 are calculated, and gradient synchronization is performed after the gradients of weights 86 to 160 are calculated. The network is divided into two shards. Therefore, gradient synchronization needs to be performed twice. The sample code is as follows:

```python
import os
import mindspore as ms
from mindspore.communication import init

device_id = int(os.getenv('DEVICE_ID', '0'))
rank_size = int(os.getenv('RANK_SIZE', '1'))
rank_id = int(os.getenv('RANK_ID', '0'))

# init context
ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=device_id)
if rank_size > 1:
    ms.set_auto_parallel_context(device_num=rank_size, parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                gradients_mean=True)
    ms.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
    init()
```

For details, see [Cluster Performance Profiling](https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_of_cluster.html).

### Data Processing Performance Tuning

The performance jitter of a single step and the empty data queue for a period of time are caused by the poor performance of the data preprocessing part. As a result, the data processing speed cannot keep up with the iteration speed of a single step. The two symptoms usually occur in pairs.

When the data processing speed is slow, the empty queue is gradually consumed from the beginning when the queue is full. The training process starts to wait for the empty queue to fill in data. Once new data is filled in, the network continues single-step training. Because no queue is used as the buffer for data processing, the performance jitter of data processing is directly reflected by the performance of a single step. Therefore, the performance jitter of a single step is also caused.

For details about data performance problems, see [Data Preparation Performance Analysis](https://www.mindspore.cn/mindinsight/docs/en/master/performance_profiling_ascend.html#data-preparation-performance-analysis) of MindSpore Insight. This describes common data performance problems and solutions.

For more performance debugging methods, see [Performance Optimization](https://www.mindspore.cn/tutorials/experts/en/master/optimize/execution_opt.html).
