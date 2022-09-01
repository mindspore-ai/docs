# 调试调优

## 功能调试

在网络的迁移过程，建议优先使用PYNATIVE模式进行调试，在PYNATIVE模式下可以进行debug，日志打印也比较友好。在调试ok后转成图模式运行，图模式在执行性能上会更友好，也可以找到一些在编写网络中的问题，比如使用了三方的算子导致梯度截断。
详情请参考 [功能调试](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/function_debug.html)。

## 精度调试

精度调试的过程基本可以分为以下过程：

### 1.检查参数

这部分包含检查所有参数和可训练参数的数量，检查所有参数的shape。

#### MindSpore获取参数方法

MindSpore可训练的参数和不可训练的参数都用`Parameter`。

```python
net = Net()
# 获取所有参数
all_parameter = []
for item in net.get_parameters():
    all_parameter.append(item)
    print(item.name, item.data.shape)
print(f"all parameter numbers: {len(all_parameter)}")

# 获取可训练的参数
trainable_params = net.trainable_params()
for item in trainable_params:
    print(item.name, item.data.shape)
print(f"trainable parameter numbers: {len(trainable_params)}")
```

#### PyTorch获取参数方法

PyTorch可训练的参数用`Parameter`，不可训练的参数`Parameter`的`requires_grad=False`或使用`buffer`。

```python
net = Net()
all_parameter = []
trainable_params = []
# 获取网络里的参数
for name, item in net.named_parameters():
    if item.requires_grad:
        trainable_params.append(item)
    all_parameter.append(item)
    print(name, item.shape)

for name, buffer in net.named_buffers():
    all_parameter.append(buffer)
    print(name, buffer.shape)
print(f"all parameter numbers: {len(all_parameter)}")
print(f"trainable parameter numbers: {len(trainable_params)}")
```

MindSpore和PyTorch的参数除了BatchNorm区别大一点，其他都差不多。注意MindSpore里没有`num_batches_tracked`的对应，实际使用时这个参数可以用优化器里的`global_step`替代。

| MindSpore | PyTorch |
| --------- | --------|
| gamma | weight |
| beta | bias |
| moving_mean | running_mean |
| moving_variance | running_var |
| 无 | num_batches_tracked |

### 2.模型验证

由于模型算法的实现是和框架没有关系的，训练好的参数可以先转换成MindSpore的[checkpoint](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/save_load.html)文件加载到网络中进行推理验证。

整个模型验证的流程请参考[resnet网络迁移](sample_code.md)。

### 3.推理验证

确认模型结构完全一致后，最好再做一次推理验证。整个推理过程除了模型外还有数据集和metrics，当推理结果不一致时，可以采用控制变量法，逐步排除问题。

整个推理验证的流程请参考[resnet网络迁移](sample_code.md)。

### 4.训练精度

当完成了推理验证后，我们基本可以确定基础模型，数据处理和metrics计算没有问题。此时如果训练的精度还是有问题时怎么进行排查呢？

- 加loss scale，在Ascend上因为Conv、Sort、TopK等算子只能是float16的，MatMul由于性能问题最好也是float16的，所以建议Loss scale操作作为网络训练的标配。

    ```python
    import mindspore as ms
    from mindspore import nn
    # Model
    loss_scale_manager = ms.FixedLossScaleManager(cfg.loss_scale) # 静态loss scale
    # loss_scale_manager = ms.DynamicLossScaleManager()   # 动态loss scale

    # 1. 一般流程
    model = ms.Model(network=net, loss_fn=loss, optimizer=opt, amp_level="O2", loss_scale_manager=loss_scale_manager)

    # 2. 自已包装正向网路和loss函数
    net.to_float(ms.float16)
    loss.to_float(ms.float32)
    net_with_loss = nn.WithLossCell(net, loss)
    # 用Model的混合精度最好要有loss_fn，否则loss部分会使用float16计算，容易溢出
    model = ms.Model(network=net_with_loss, optimizer=opt, loss_scale_manager=loss_scale)

    # 3. 自己包装训练流程
    scale_sense = nn.FixedLossScaleUpdateCell(cfg.loss_scale) # 静态loss scale
    # scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale, scale_factor=2, scale_window=1000) # 动态loss scale
    train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer=opt, scale_sense=scale_sense)
    model = ms.Model(network=train_net)
    ```

- 排查是否溢出，添加loss scale时，默认会加上溢出检测，可以将是否溢出的结果进行监测，如果持续溢出的话建议优先排查为什么溢出，建议使用MindInsight的[调试器](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/debugger.html)或者[dump数据](https://mindspore.cn/tutorials/experts/zh-CN/master/debug/dump.html)。

    ```python
    train_net.set_train()
    iterator = dataset.create_tuple_iterator()
    for i, data in enumerate(iterator):
        loss, overflow, scaling_sens = train_net(*data)
        print("step: {}, loss: {}, overflow:{}, scale:{}".format(i, loss, overflow, scaling_sens))
    ```

- 排查优化器和loss，整个训练过程除了模型、数据集外新加的部分只有优化器和loss，训练有问题时需要重点排查，尤其是loss，出现问题的概率较大。
- 多卡确认是否加seed保证多卡初始化一致，[自定义训练](model_development/training_and_gradient.md#自定义训练Cell)确认是否进行梯度聚合。

    ```python
    import mindspore as ms
    ms.set_seed(1) # 会固定MindSpore的，numpy的，dataset的随机种子，API内部的需要在API属性设置
    ```

- 排查数据处理，通过可视化等方法查看数据处理是否符合预期，重点查看数据shuffle，是否有数据不匹配的情况。

更多精度调试策略请参考[精度调试](https://mindspore.cn/mindinsight/docs/zh-CN/master/accuracy_problem_preliminary_location.html)。

## 性能调优

性能优化方向主要包含：

1. 算子性能优化
2. 框架使能性能优化
3. 多机同步性能优化
4. 数据处理性能优化

可以参考[resnet网络迁移](sample_code.md)串通整个过程。

> 有的网络很大或者有很多[流程控制语句](https://mindspore.cn/tutorials/zh-CN/master/advanced/network/control_flow.html)，这种情况在图模式下编译会很慢。在性能调优过程请区分图编译和网络执行，本节主要介绍网络执行阶段的性能调优策略，如果确认是图编译慢请尝试[算子增量编译](https://mindspore.cn/tutorials/experts/zh-CN/master/debug/op_compilation.html)或者联系 [MindSpore社区](https://gitee.com/mindspore/mindspore/issues) 反馈。

### 算子性能优化

#### 算子性能问题

单算子耗时久、对于同一种算子在不同shape或者不同 datatype 下性能差异较大的情况主要是由算子性能问题引起，通常有以下解决思路：

1. 使用计算量更小的数据类型。例如，同一个算子在 float16 和 float32 下精度无明显差别，可使用计算量更小的 float16 格式。
2. 使用算法相同的其他算子规避。
3. Ascend环境上注意16对齐。由于昇腾芯片的设计，在AICore上的计算最好是16对齐的(shape中的每一维都是16的倍数)。
4. [算子调优](https://mindspore.cn/tutorials/experts/zh-CN/master/debug/auto_tune.html)。

如果您发现有性能较差的算子时，建议联系 [MindSpore社区](https://gitee.com/mindspore/mindspore/issues) 反馈，我们确认为性能问题后会及时优化。

### 框架使能性能优化

#### 使用静态图模式

MindSpore一般在静态图模式下比PYNATIVE模式下快很多，最好能在静态图模式下进行训练和推理，具体原理请参考[动静态图结合](https://www.mindspore.cn/docs/zh-CN/master/design/dynamic_graph_and_static_graph.html)。

#### on-device执行

MindSpore提供了一种[on-device执行](https://www.mindspore.cn/docs/zh-CN/master/design/on_device.html)的方法将数据处理和网络在device上的执行并行起来，只需要在`model.train`中设置`dataset_sink_mode=True`即可，注意这个配置默认是`True`当打开这个配置时，一个epoch只会返回一个网络的结果，当进行调试时建议先将这个值改成`False`。

#### 使用自动混合精度

混合精度训练方法是通过混合使用单精度和半精度数据格式来加速深度神经网络训练的过程，同时保持了单精度训练所能达到的网络精度。混合精度训练能够加速计算过程，同时减少内存使用和存取，并使得在特定的硬件上可以训练更大的模型或 batch size。

具体可参考 [混合精度教程](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)。

#### 使能图算融合

图算融合是 MindSpore 特有的网络性能优化技术。它可以通过自动分析和优化现有网络计算图逻辑，并结合目标硬件能力，对计算图进行计算化简和替代、算子拆分和融合、算子特例化编译等优化，以提升设备计算资源利用率，实现对网络性能的整体优化。相比传统优化技术，图算融合具有多算子跨边界联合优化、与算子编译跨层协同、基于Polyhedral的算子即时编译等独特优势。另外，图算融合只需要用户打开对应配置后，整个优化过程即可自动完成，不需要网络开发人员进行其它额外感知，使得用户可以聚焦网络算法实现。

图算融合的适用场景包括：对网络执行时间具有较高性能要求的场景；通过拼接基本算子实现自定义组合算子，并希望对这些基本算子进行自动融合，以提升自定义组合算子性能的场景。

具体可参考 [图算融合教程](https://www.mindspore.cn/docs/zh-CN/master/design/graph_fusion_engine.html)。

#### 其他

转换算子过多（TransData、Cast类算子）且耗时明显时，如果是我们手动加入的Cast算子，可分析其必要性，如果对精度没有影响，可去掉冗余的Cast、TransData算子。

如果是MindSpore自动生成的转换算子过多，可能是MindSpore框架针对某些特殊情况没有充分优化，可联系 [MindSpore社区](https://gitee.com/mindspore/mindspore/issues) 反馈。

[动态shape场景](analysis_and_preparation.md#动态shape)目前需要不断的编图，可能会造成端到端的训练时间较长，建议优先[规避动态shape](model_development/model_and_loss.md#动态shape规避策略)。

### 多机同步性能优化

当进行分布式训练时，在一个Step的训练过程中，完成前向传播和梯度计算后，各个机器开始进行AllReduce梯度同步，AllReduce同步时间主要受权重数量、机器数量影响，对于越复杂、机器规模越大的网络，其 AllReduce 梯度更新时间也越久，此时我们可以进行AllReduce 切分来优化这部分耗时。

正常情况下，AllReduce 梯度同步会等所有反向算子执行结束，也就是对所有权重都计算出梯度后再一次性同步所有机器的梯度，而使用AllReduce切分后，我们可以在计算出一部分权重的梯度后，就立刻进行这部分权重的梯度同步，这样梯度同步和剩余算子的梯度计算可以并行执行，也就隐藏了这部分 AllReduce 梯度同步时间。切分策略通常是手动尝试，寻找一个最优的方案（支持切分大于两段）。
以 [ResNet50网络](https://gitee.com/mindspore/models/blob/master/official/cv/resnet/train.py) 为例，该网络共有 160  个 权重，  [85, 160] 表示第 0 至 85个权重计算完梯度后立刻进行梯度同步，第 86 至 160 个 权重计算完后再进行梯度同步，这里共切分两段，因此需要进行两次梯度同步。代码实现如下：

```python
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

更多请参考[集群性能调试](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_of_cluster.html)以及[并行训练执行分析](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_of_parallel_training.html)。

### 数据处理性能优化

单Step性能抖动、数据队列一段时间内持续为空的情况都是由于数据预处理部分性能较差，使得数据处理速度跟不上单Step迭代速度导致，这两个现象通常成对出现。

当数据处理速度较慢时，队列从最开始的满队列情况逐渐消耗为空队列，训练进程会开始等待空队列填入数据，一旦有新的数据填入，网络才会继续进行单Step训练。由于数据处理没有队列作为缓冲，数据处理的性能抖动直接体现在单Step的性能上，因此还会造成单Step性能抖动。

关于数据的性能问题，可以参考 MindInsight 组件的 [数据准备性能分析](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/performance_profiling_ascend.html#数据准备性能分析)，其给出了数据性能的常见问题及解决方法。

更多性能调试方法请参考[性能调优](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/performance_optimization.html)。
