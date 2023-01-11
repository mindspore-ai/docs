# Debugging and Tuning

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/migration_guide/debug_and_tune.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Function Debugging

During network migration, you are advised to use the PyNative mode for debugging. In PyNative mode, you can perform debugging, and log printing is user-friendly. After the debugging is complete, the graph mode is used. The graph mode is more user-friendly in execution performance. You can also find some problems in network compilation. For example, gradient truncation caused by third-party operators.
For details, see [Function Debugging](https://mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/debug/function_debug.html).

## Accuracy Debugging

The accuracy debugging process is as follows:

### 1. Checking Parameters

This part includes checking all parameters and the number of trainable parameters, and checking the shape of all parameters.

#### Obtaining MindSpore Parameters

`Parameter` is used for MindSpore trainable and untrainable parameters.

```python
from mindspore import nn

class msNet(nn.Cell):
    def __init__(self):
        super(msNet, self).__init__()
        self.fc = nn.Dense(1, 1, weight_init='normal')
    def construct(self, x):
        output = self.fc(x)
        return output

msnet = msNet()
# Obtain all parameters.
all_parameter = []
for item in msnet.get_parameters():
    all_parameter.append(item)
    print(item.name, item.data.shape)
print(f"all parameter numbers: {len(all_parameter)}")

# Obtain trainable parameters.
trainable_params = msnet.trainable_params()
for item in trainable_params:
    print(item.name, item.data.shape)
print(f"trainable parameter numbers: {len(trainable_params)}")
```

```text
    fc.weight (1, 1)
    fc.bias (1,)
    all parameter numbers: 2
    fc.weight (1, 1)
    fc.bias (1,)
    trainable parameter numbers: 2
```

#### Obtaining PyTorch Parameters

`Parameter` is used for PyTorch trainable parameters, and `requires_grad=False` or `buffer` is used for PyTorch untrainable parameters.

```python
from torch import nn

class ptNet(nn.Module):
    def __init__(self):
        super(ptNet, self).__init__()
        self.fc = nn.Linear(1, 1)
    def construct(self, x):
        output = self.fc(x)
        return output


ptnet = ptNet()
all_parameter = []
trainable_params = []
# Obtain network parameters.
for name, item in ptnet.named_parameters():
    if item.requires_grad:
        trainable_params.append(item)
    all_parameter.append(item)
    print(name, item.shape)

for name, buffer in ptnet.named_buffers():
    all_parameter.append(buffer)
    print(name, buffer.shape)
print(f"all parameter numbers: {len(all_parameter)}")
print(f"trainable parameter numbers: {len(trainable_params)}")
```

```text
    fc.weight torch.Size([1, 1])
    fc.bias torch.Size([1])
    all parameter numbers: 2
    trainable parameter numbers: 2
```

The parameters of MindSpore and PyTorch are similar except BatchNorm. Note that MindSpore does not have parameters corresponding to `num_batches_tracked`. You can replace this parameter with `global_step` in the optimizer.

| MindSpore | PyTorch |
| --------- | --------|
| gamma | weight |
| beta | bias |
| moving_mean | running_mean |
| moving_variance | running_var |
| -| num_batches_tracked |

### 2. Model Verification

The implementation of the model algorithm is irrelevant to the framework. The trained parameters can be converted into the [checkpoint](https://www.mindspore.cn/tutorials/en/r2.0.0-alpha/beginner/save_load.html) file of MindSpore and loaded to the network for inference verification.

For details about the model verification process, see [ResNet Network Migration](https://www.mindspore.cn/docs/en/r2.0.0-alpha/migration_guide/sample_code.html#model-validation).

### 3. Inference Verification

After confirming that the model structures are the same, you are advised to perform inference verification again. In addition to models, the entire inference process also involves datasets and metrics. When the inference results are inconsistent, you can use the control variable method to gradually rectify the fault.

For details about the inference verification process, see [ResNet Network Migration](https://www.mindspore.cn/docs/en/r2.0.0-alpha/migration_guide/sample_code.html#inference-process).

### 4. Training Accuracy

After the inference verification is complete, the basic model, data processing, and metrics calculation are normal. If the training accuracy is still abnormal, how do we locate the fault?

- Add loss scale. On Ascend, operators such as Conv, Sort, and TopK can only be float16. MatMul is recommended to be float16 due to performance problems. Therefore, it is recommended that loss scale be used as a standard configuration for network training.

The list of operators only supports float16 on Ascend:

| type | operators |
| ------  | ------ |
| Pool    | AdaptiveMaxPool2D，AvgPool3D，AvgPool，MaxPool，MaxPoolWithArgmax，Pooling |
| RNN     | LSTM，DynamicRNN，GRUV2 |
| Conv    | Conv2D，Conv2DTranspose，Conv3D，Conv3DTranspose，DepthwiseConv2dNative |
| Matmul (float32 is too slow and needs to be cast to float16) | MatMul，BatchMatMul |
| Sort | Sort，TopK |
| Others | BoundingBoxEncode，ExtractImagePatches，ExtractVolumePatches，FusedDbnDw，IOU，NewIm2Col，NMSWithMask |

```python
import mindspore as ms
from mindspore import nn
# Model
loss_scale_manager = ms.FixedLossScaleManager(drop_overflow_update=False) # Static loss scale
# loss_scale_manager = ms.DynamicLossScaleManager()   # Dynamic loss scale

# 1. General process
loss = nn.MSELoss()
opt = nn.Adam(params=msnet.trainable_params(), learning_rate=0.01)
model = ms.Model(network=msnet, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager)

# 2. Self-packaged forward network and loss function
msnet.to_float(ms.float16)
loss.to_float(ms.float32)
net_with_loss = nn.WithLossCell(msnet, loss)
# It is recommended that loss_fn be used for the mixed precision of the model. Otherwise, float16 is used for calculation of the loss part, which may cause overflow.
model = ms.Model(network=net_with_loss, optimizer=opt)

# 3. Self-packaged training process
scale_sense = nn.FixedLossScaleUpdateCell(1)#(config.loss_scale) # Static loss scale
# scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=config.loss_scale,
#                                             scale_factor=2, scale_window=1000) # Dynamic loss scale
train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer=opt, scale_sense=scale_sense)
model = ms.Model(network=train_net)
```

- Check whether overflow occurs. When loss scale is added, overflow detection is added by default to monitor the overflow result. If overflow occurs continuously, you are advised to use the [debugger](https://www.mindspore.cn/mindinsight/docs/en/r2.0.0-alpha/debugger.html) or [dump data](https://mindspore.cn/tutorials/experts/en/r2.0.0-alpha/debug/dump.html) of MindInsight to check why overflow occurs.

```python
import numpy as np
from mindspore import dataset as ds

def get_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)


def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size, drop_remainder=True)
    input_data = input_data.repeat(repeat_size)
    return input_data

train_net.set_train()
dataset = create_dataset(1600)
iterator = dataset.create_tuple_iterator()
for i, data in enumerate(iterator):
    loss, overflow, scaling_sens = train_net(*data)
    print("step: {}, loss: {}, overflow:{}, scale:{}".format(i, loss, overflow, scaling_sens))
```

```text
    step: 0, loss: 138.42825, overflow:False, scale:1.0
    step: 1, loss: 118.172104, overflow:False, scale:1.0
    step: 2, loss: 159.14542, overflow:False, scale:1.0
    step: 3, loss: 150.65671, overflow:False, scale:1.0
    ... ...
    step: 97, loss: 69.513245, overflow:False, scale:1.0
    step: 98, loss: 51.903114, overflow:False, scale:1.0
    step: 99, loss: 42.250656, overflow:False, scale:1.0
```

- Check the optimizer, loss, and parameter initialization. In addition to the model and dataset, only the optimizer, loss, and parameter initialization are added in the entire training process. If the training is abnormal, check the optimizer, loss, and parameter initialization. Especially for loss and parameter initialization, there is a high probability that the problem occurs.
- Check whether to add seeds for multiple devices to ensure that the initialization of multiple SIM cards is consistent. Determine whether to perform gradient aggregation during [customized training](https://www.mindspore.cn/docs/en/r2.0.0-alpha/migration_guide/model_development/training_and_gradient.html#customizing-training-cell).

```python
import mindspore as ms
ms.set_seed(1) # The random seeds of MindSpore, NumPy, and dataset are fixed. The random seed of the API needs to be set in the API attribute.
```

- Check whether the data processing meets the expectation through visualization. Focus on data shuffle and check whether data mismatch occurs.

For details about more accuracy debugging policies, see [Accuracy Debugging](https://mindspore.cn/mindinsight/docs/en/r2.0.0-alpha/accuracy_problem_preliminary_location.html).

## Performance Tuning

The performance tuning directions are as follows:

1. Operator performance tuning
2. Framework enabling performance tuning
3. Multi-Node synchronization performance tuning
4. Data processing performance tuning

For details, see [ResNet Network Migration](https://www.mindspore.cn/docs/en/r2.0.0-alpha/migration_guide/sample_code.html).

> Some networks are large or there are many [process control statements](https://mindspore.cn/tutorials/experts/en/r2.0.0-alpha/network/control_flow.html). In this case, the build is slow in graph mode. During performance tuning, distinguish graph build from network execution. This section describes the performance tuning policies in the network execution phase. If graph build is slow, try [incremental operator build](https://mindspore.cn/tutorials/experts/en/r2.0.0-alpha/debug/op_compilation.html) or contact [MindSpore community](https://gitee.com/mindspore/mindspore/issues) for feedback.

### Operator Performance Tuning

#### Poor Operator Performance

If a single operator takes a long time and the performance of the same operator varies greatly in different shapes or data types, the problem is caused by the operator performance. The solution is as follows:

1. Use data types with less computational workload. For example, if there is no obvious difference between the precision of the same operator in float16 and float32 modes, you can use the float16 format with less calculation workload.
2. Use other operators with the same algorithm to avoid this problem.
3. Pay attention to 16-alignment in the Ascend environment. Due to the design of the Ascend AI Processors, it is recommended that the calculation on the AI core be 16-alignment (each dimension in the shape is a multiple of 16).
4. [Operator Tuning](https://mindspore.cn/tutorials/experts/en/r2.0.0-alpha/debug/auto_tune.html).

If you find an operator with poor performance, you are advised to contact [MindSpore community](https://gitee.com/mindspore/mindspore/issues) for feedback. We will optimize the operator in time after confirming that the problem is caused by poor performance.

### Framework Enabling Performance Tuning

#### Using the Static Graph Mode

Generally, MindSpore in static graph mode is much faster than that in PyNative mode. It is recommended that training and inference be performed in static graph mode. For details, see [Combination of Dynamic and Static Graphs](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/design/dynamic_graph_and_static_graph.html).

#### On-device Execution

MindSpore provides an [on-device execution method](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/design/overview.html) to concurrently process data and execute the network on the device. You only need to set `dataset_sink_mode=True` in `model.train`. Note that this configuration is `True` by default. When this configuration is enabled, one epoch returns the result of only one network. You are advised to change the value to `False` during debugging.

#### Using Automatic Mixed Precision

The mixed precision training method accelerates the deep neural network training process by mixing the single-precision floating-point data format and the half-precision floating-point data format without compromising the network accuracy. Mixed precision training can accelerate the computing process, reduce memory usage and retrieval, and enable a larger model or batch size to be trained on specific hardware.

For details, see [Mixed Precision Tutorial](https://www.mindspore.cn/tutorials/zh-CN/r2.0.0-alpha/advanced/mixed_precision.html).

#### Enabling Graph Kernel Fusion

Graph kernel fusion is a unique network performance optimization technology of MindSpore. It can automatically analyze and optimize the logic of existing network computational graphs, simplify and replace computational graphs, split and fuse operators, and build operators in a special way based on the target hardware capability to improve the computing resource utilization of devices and optimize the overall network performance. Compared with traditional optimization technologies, the graph kernel fusion technology has unique advantages, such as joint optimization of multiple operators across boundaries, cross-layer collaboration with operator compilation, and real-time compilation of operators based on Polyhedral. In addition, the entire optimization process of graph kernel fusion can be automatically completed after users enable the corresponding configuration. Network developers do not need to perform extra perception, so that users can focus on network algorithm implementation.

Graph kernel fusion applies to scenarios that have high requirements on network execution time. Basic operators are combined to implement customized combination operators and these basic operators are automatically fused to improve the performance of the customized combination operators.

For details, see [Graph Kernel Fusion Tutorial](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/design/graph_fusion_engine.html).

#### Others

If there are too many conversion operators (TransData and Cast operators) and the conversion takes a long time, analyze the necessity of the manually added Cast operator. If the accuracy is not affected, delete the redundant Cast and TransData operators.

If there are too many conversion operators automatically generated by MindSpore, the MindSpore framework may not be fully optimized for some special cases. In this case, contact [MindSpore community](https://gitee.com/mindspore/mindspore/issues) for feedback.

In [dynamic shape scenario](https://www.mindspore.cn/docs/en/r2.0.0-alpha/migration_guide/analysis_and_preparation.html), continuous graph build is required, which may cause a long end-to-end training time. You are advised to [avoid dynamic shape](https://www.mindspore.cn/docs/en/r2.0.0-alpha/migration_guide/model_development/model_and_loss.html).

### Multi-Node Synchronization Performance Tuning

During distributed training, after forward propagation and gradient calculation are complete in a step training process, each machine starts to perform AllReduce gradient synchronization. The AllReduce synchronization time is mainly affected by the number of weights and machines. For a more complex network with a larger machine scale, the AllReduce gradient update time is longer. In this case, you can perform AllReduce segmentation to reduce the time consumption.

In normal cases, AllReduce gradient synchronization waits until all backward operators are executed. That is, after the gradient of all gradients is calculated, the gradients of all machines are synchronized at a time. After AllReduce segmentation is used, the gradients of some weights can be calculated, gradient synchronization of this part of weights is immediately performed. In this way, gradient synchronization and gradient calculation of remaining operators can be performed concurrently, and this part of AllReduce gradient synchronization time is hidden. The shard strategy is usually manually tried to find an optimal solution (more than two shards are supported).
The [ResNet-50](https://gitee.com/mindspore/models/blob/r2.0/official/cv/ResNet/train.py) is used as an example. The network has 160 weights. [85, 160] indicates that gradient synchronization is performed immediately after the gradients of weights 0 to 85 are calculated, and gradient synchronization is performed after the gradients of weights 86 to 160 are calculated. The network is divided into two shards. Therefore, gradient synchronization needs to be performed twice. The sample code is as follows:

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

For details, see [Cluster Performance Profiling](https://www.mindspore.cn/mindinsight/docs/en/r2.0.0-alpha/performance_profiling_of_cluster.html).

### Data Processing Performance Tuning

The performance jitter of a single step and the empty data queue for a period of time are caused by the poor performance of the data preprocessing part. As a result, the data processing speed cannot keep up with the iteration speed of a single step. The two symptoms usually occur in pairs.

When the data processing speed is slow, the empty queue is gradually consumed from the beginning when the queue is full. The training process starts to wait for the empty queue to fill in data. Once new data is filled in, the network continues single-step training. Because no queue is used as the buffer for data processing, the performance jitter of data processing is directly reflected by the performance of a single step. Therefore, the performance jitter of a single step is also caused.

For details about data performance problems, see [Data Preparation Performance Analysis](https://www.mindspore.cn/mindinsight/docs/en/r2.0.0-alpha/performance_profiling_ascend.html#data-preparation-performance-analysis) of MindInsight. This describes common data performance problems and solutions.

For more performance debugging methods, see [Performance Tuning](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/debug/performance_optimization.html).
