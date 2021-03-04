# 网络迁移调试实例

<!-- TOC -->

- [网络迁移调试实例](#网络迁移调试实例)
    - [对标网络分析与复现](#对标网络分析与复现)
        - [确定迁移目标](#确定迁移目标)
        - [复现迁移目标](#复现迁移目标)
        - [复现单Step结果](#复现单step结果)
    - [脚本开发](#脚本开发)
        - [脚本开发前分析](#脚本开发前分析)
        - [数据预处理](#数据预处理)
        - [子网开发](#子网开发)
        - [其他模块](#其他模块)
        - [超参对比](#超参对比)
    - [精度调试](#精度调试)
        - [训练](#训练)
        - [单机训练](#单机训练)
        - [多机训练精度调优](#多机训练精度调优)
    - [性能调优](#性能调优)
        - [分析Profiling数据](#分析profiling数据)
        - [常见问题及相应优化方法](#常见问题及相应优化方法)
            - [MindData性能问题](#minddata性能问题)
            - [多机同步性能问题](#多机同步性能问题)
            - [算子性能问题](#算子性能问题)
            - [框架性能问题](#框架性能问题)
            - [其他通用优化方法](#其他通用优化方法)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/migration_guide/source_zh_cn/sample_code.md" target="_blank"><img src="./_static/logo_source.png"></a>

本章将结合用例来介绍网络迁移的基本步骤、常用工具、定位问题的思路及解决方法。

## 对标网络分析与复现

### 确定迁移目标

网络迁移的第一步是确定迁移目标，既先找到一个合适的、可达成的标准，通常一个深度神经网络的交付目标包括以下四个部分：

1. 网络实现：这是迁移目标中最基本的部分，有时同一个神经网络有不同的版本、同一个版本有不同的实现方式或者在相同的神经网络下使用不同的超参，这些差别会对最终的收敛精度和性能造成一定影响。通常，我们以神经网络作者本身的实现为准，也可以参考不同框架（例如TensorFlow、PyTorch等）的官方实现或其他主流开源工具箱（例如 MMDetection）。
2. 数据集：相同的神经网络和参数，在不同的数据集上往往差别很大，因此我们需要确认迁移网络所使用的数据集。一些数据集的数据会频繁更新，确定数据集时需要注意数据集的版本、训练数据和测试数据划分比例等问题。
3. 收敛精度：不同的框架、不同的GPU型号、是否为分布式训练等因素会对精度有所影响，在确定迁移目标时需要分析清楚对标的框架、硬件等信息。
4. 训练性能：和收敛精度相同，训练性能主要受框架、GPU本身和是否为分布式训练因素影响。

### 复现迁移目标

网络迁移目标确定完成后，接下来要做的就是复现这些指标。复现标杆数据对后续精度和性能调优十分重要，当我们在 MindSpore开发的网络和对标脚本有精度/性能差距时，很多时候都是以标杆数据作为基准，一步一步地分析迁移脚本和对标脚本的差别，如果对标脚本无法复现指标，那我们以此为基准开发的MindSpore脚本就很难达到迁移目标。复现迁移指标时，不仅要复现训练阶段，推理阶段也同样重要。

需要注意的是对于部分网络，使用相同的硬件环境和脚本训练，最终达到的收敛精度和性能也可能与原作者提出的结果有细微差别，这属于正常的波动范围，我们在迁移网络时要把这种波动考虑在内。

### 复现单Step结果

复现单Step结果主要是为了接下来的脚本开发和网络调优。对于复杂的神经网络，完整的训练需要耗时几天甚至几个月，如果仅以最终的训练精度和结果做参考，会极大地降低开发效率。因此，我们需要提前复现单Step的运行结果，并以此为对照展开后续的开发工作。

## 脚本开发

### 脚本开发前分析

在开始真正的开发脚本前，需要进行对标脚本分析。脚本分析的目的是识别出MindSpore与对标框架相比缺失的算子或功能。MindSpore已支持绝大多数常用[功能](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/index.html)和[算子](https://www.mindspore.cn/doc/programming_guide/zh-CN/master/operator_list.html)。MindSpore既支持动态图（PyNative）模式，又支持静态图（Graph) 模式，动态图模式灵活、易于调试，因此动态图模式主要用于网络调试，静态图模式性能好，主要用于整网训练，在分析缺失算子和功能时，要分别分析这两种模式。

如果发现有缺失的算子和功能，首先可考虑基于当前算子或功能来组合出缺失的算子和功能，对于主流的CV和NLP类网络，新的缺失算子一般都可以通过组合已有算子的方式来解决。

组合的算子可以通过Cell的方式实现，在MindSpore中，[nn类算子](https://gitee.com/mindspore/mindspore/tree/master/mindspore/nn) 就是通过这种方式实现的。例如下面的`ReduceSumExp`算子，它是由已有的`Exp`、`ReduceSum`、`Log`小算子组合而成：

```python
class ReduceLogSumExp(Cell):
    r"""
    Reduces a dimension of a tensor by calculating exponential for all elements in the dimension,
    then calculate logarithm of the sum.

    The dtype of the tensor to be reduced is number.

    .. math::

        ReduceLogSumExp(x) = \log(\sum(e^x))

    Args:
        axis (Union[int, tuple(int), list(int)]) - The dimensions to reduce. Default: (), reduce all dimensions.
            Only constant value is allowed.
        keep_dims (bool): If True, keep these reduced dimensions and the length is 1.
            If False, don't keep these dimensions.
            Default : False.

    Inputs:
        - **x** (Tensor) - The input tensor. With float16 or float32 data type.

    Outputs:
        Tensor, has the same dtype as the `x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the sum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> op = nn.ReduceLogSumExp(1, keep_dims=True)
        >>> output = op(input_x)
        >>> print(output.shape)
        (3, 1, 5, 6)
    """

    def __init__(self, axis, keep_dims=False):
        super(ReduceLogSumExp, self).__init__()
        validator.check_value_type('axis', axis, [int, list, tuple], self.cls_name)
        validator.check_value_type('keep_dims', keep_dims, [bool], self.cls_name)
        self.axis = axis
        self.exp = P.Exp()
        self.sum = P.ReduceSum(keep_dims)
        self.log = P.Log()

    def construct(self, x):
        exp = self.exp(x)
        sumexp = self.sum(exp, self.axis)
        logsumexp = self.log(sumexp)
        return logsumexp
```

如果缺失的功能和算子无法规避，或者组合算子性能较差，严重影响网络的训练和推理，可联系[MindSpore社区](https://gitee.com/mindspore/mindspore/issues) 反馈，我们会有专门的工作人员为您解决。

### 数据预处理

要理解一个神经网络的实现，首先要清楚网络的输入数据，因此，数据预处理是脚本开发的第一个环节。MindSpore设计了一个专门进行数据处理的模块 - MindData，使用MindData进行数据预处理主要包括以下几个步骤：

1. 传入数据路径，读取数据文件。
2. 解析数据。
3. 数据处理（如常见数据切分、shuffle、数据增强等操作）。
4. 数据分发（以batch_size为单位分发数据，分布式训练涉及多机分发）。

在读取和解析数据过程中，MindSpore提供了一种更友好的数据格式 - [MindRecord](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/convert_dataset.html)。用户可以将常规格式的数据集转换为 MindSpore数据格式，即MindRecord，从而方便地加载到MindSpore中进行训练。同时，MindSpore在部分场景做了性能优化，使用MindSpore数据格式可以获得更好的性能。

数据处理通常是数据准备中最耗时的阶段，大部分对数据的操作都被包含在这一步骤里，例如CV类网络中的Resize、Rescale、Crop等操作。MindSpore提供了一套常用的数据处理集成接口，用户可以不用自己实现而直接调用这些接口，这些集成接口不仅可以提升用户的易用性，还可以提升数据预处理的性能，减少训练过程中数据准备的耗时。具体可以参考<https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/optimize_data_processing.html>。

在数据分发环节，MindData提供了极为简洁的API，可以通过直接调用batch、repeat等操作完成数据的batch组合、重复等操作。

当完成以上4个步骤后，我们使用MindSpore脚本和对标脚本处理数据集后，可以得到完全相同的数据（如果有引入随机情况的操作需要去除）。

以[ResNet50网络](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/resnet/src/dataset.py)和CIFAR-10数据预处理为例：

```python
def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", distribute=False):
    """
    create a train or evaluate cifar10 dataset for resnet50
    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: Ascend
        distribute(bool): data for distribute or not. Default: False

    Returns:
        dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if distribute:
            init()
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            device_num = 1
    if device_num == 1:
        # 单机训练
        # num_paralel_workers: 并行数据处理的并行度
        # shuffle: 是否打乱原数据顺序
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        # 多机训练 (num_parallel_workers, shuffle 意义同上)
        # num_shards: 分布式训练的总机器数量，即数据要被切分的总数量
        # shard_id: 当前机器在所有训练机器的序列，该机器只能获取第shard_id份数据
        data_set = ds.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                     num_shards=device_num, shard_id=rank_id)

    # define map operations
    trans = []
    if do_train:
        trans += [
            # 声明要使用的MindData内置数据处理接口，提升处理性能（注意，这两个操作会引入随机情况）
            C.RandomCrop((32, 32), (4, 4, 4, 4)),
            C.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        # 声明要使用的MindData内置数据处理接口，提升处理性能
        C.Resize((224, 224)),
        C.Rescale(1.0 / 255.0, 0.0),
        C.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        C.HWC2CHW()
    ]

    type_cast_op = C2.TypeCast(mstype.int32)

    # 调用已定义好的内置预处理接口
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)

    # 将处理好的数据已经batch操作
    data_set = data_set.batch(batch_size, drop_remainder=True)
    # 将已做过batch的数据进行repeat操作，repeat_num代表数据被重复的次数
    data_set = data_set.repeat(repeat_num)

    return data_set
```

### 子网开发

通常子网开发包含两个部分：训练子网和loss子网，其中训练子网可根据网络的复杂程度决定是否继续划分。直接开发一个大型的神经网络脚本可能会让我们无从下手，因此，我们可以将网络中不同模块或子模块作为一个个子网抽离出来单独开发，这样可以保证各个子网并行开发，互相不受干扰。子网开发完成后，还可以固定子网输入和权重，与对标脚本的子网代码形成对比，作为后续网络开发的测试用例。

在精度调优阶段，我们常常会遇到精度不达标的情况，这时我们会重新审视已开发的脚本并逐行排查。而使用子网方式开发脚本并形成测试用例可以高效地帮助我们的排除怀疑点，从几十个算子里寻找可疑点，要比从成百上千个算子中找可疑点轻松得多，尤其是在很多时候，同一个子网会被重复调用多次，当我们以子网为单位排查时，可以减少很多工作量。

### 其他模块

其他模块通常包括：反向构造、梯度裁剪、优化器、学习率生成等，这些模块要么本身结构单一，要么依赖已开发完成的子网结果才能和对标脚本形成对比。相比子网开发，这些模块的脚本开发难度更小一些。

### 超参对比

当各子网已经打通，最后一步要做的是和对标脚本对齐超参，保证网络结构一致。

## 精度调试

### 训练

### 单机训练

### 多机训练精度调优

## 性能调优

通常我们所指的性能调优是在固定数据集、网络规模和硬件数量的情况下提高训练性能，而通过改变数据集大小、网络规模、硬件数量来提高性能是显然的，不在本文的讨论范围内。

除非性能问题已严重阻碍了精度调试，否则性能调优一定要放在精度达标以后进行，这其中主要有两个原因：一是在定位精度问题时很多修改会影响性能，使得已经调优过的性能再次未达标，可能浪费工作量；二是性能调优时有可能引入新的精度问题，如果没有已经达标的精度作为看护，后面再定位这次引入的精度问题难度会极大的增加。

### 分析Profiling数据

分析Profiling数据是性能调优阶段必不可少的步骤，MindSpore的性能和精度调优工具[MindInsight](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/visualization_tutorials.html)提供了丰富的性能和精度调优方法，对于性能调优，最重要的信息就是Profiling数据。Profiling可以收集整网训练过程中端到端的详细性能数据，包含数据准备和迭代轨迹。在迭代轨迹中，你可以看到每个算子的起始运行时间、结束运行时间、调用次数和调用顺序等非常详细的信息，这对我们性能调优非常有帮助。生成Profiling数据的方式如下：

```python
from mindspore.profiler import Profiler
from mindspore import Model, nn, context

# 初始化context
context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=int(os.environ["DEVICE_ID"]))

# 初始化profiler，默认数据会保存在当前路径的data目录下
profiler = Profiler()

# 训练
Model.train()

# 训练结束，解析profiling数据成可读文本
profiler.analyse()
```

关于Profiling更详细的方法，可以参考[性能调优教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/performance_profiling.html)。

获取到Profiling数据后，我们可以分析出性能瓶颈阶段和算子，然后使用以下手段优化性能。

### 常见问题及相应优化方法

#### MindData性能问题

单Step性能抖动、数据队列一段时间内持续为空的情况都是由于数据预处理部分性能较差，使得数据处理速度跟不上单Step迭代速度导致，这两个现象通常成对出现。

当数据处理速度较慢时，队列从最开始的满队列情况逐渐消耗为空队列，训练进程会开始等待空队列填入数据，一旦有新的数据填入，网络才会继续进行单Step训练。由于数据处理没有队列作为缓冲，数据处理的性能抖动直接体现在单Step的性能上，因此还会造成单Step性能抖动。

#### 多机同步性能问题

当进行分布式训练时，在一个Step的训练过程中，完成前向传播和梯度计算后，各个机器开始进行AllReduce梯度同步，AllReduce同步时间主要受权重数量、机器数量影响，对于越复杂、机器规模越大的网络，其AllReduce梯度更新时间也越久，此时我们可以进行AllReduce切分来优化这部分耗时。

正常情况下，AllReduce梯度同步会等所有反向算子执行结束，也就是对所有权重都计算出梯度后再一次性同步所有机器的梯度，而使用AllReduce切分后，我们可以在计算出一部分权重的梯度后，就立刻进行这部分权重的梯度同步，这样梯度同步和剩余算子的梯度计算可以并行执行，也就隐藏了这部分AllReduce梯度同步时间。

以 [ResNet50网络](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/resnet/train.py)为例：

```python
from mindspore import context
...

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
set_algo_parameters(elementwise_op_strategy_follow=True)
if args_opt.net == "resnet50" or args_opt.net == "se-resnet50":
    # AllReduce 切分
    # [85, 160] 表示前 0-85, 86-160个parameter计算完梯度后立刻进行梯度同步。resnet50共有160个parameter，因此这里进行两次梯度同步
    # 切分策略通常是手动尝试，寻找一个最优的方案（支持切分大于两段）
    context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
else:
    # 不同的网络结构可以设置不同的切分策略
    context.set_auto_parallel_context(all_reduce_fusion_config=[180, 313])
init()
```

#### 算子性能问题

单算子耗时久、对于同一种算子在不同shape或者不同datatype下性能差异较大的情况主要是由算子性能问题引起，通常有以下两个解决思路：

1. 使用计算量更小的数据类型。例如，同一个算子在float16和float32下精度无明显差别，可使用计算量更小的float16格式。
2. 使用Ascend芯片更亲和的Format。为了充分发挥Ascend芯片的算力，我们设计了几种Ascend亲和的Format，可以尝试使用这些Format。
3. 使用算法相同的其他算子规避。

如果您发现有性能较差的算子时，建议联系 [MindSpore社区](https://gitee.com/mindspore/mindspore/issues) 反馈，我们确认为性能问题后会及时优化。

#### 框架性能问题

转换算子过多（TransData、Cast类算子）且耗时明显时，如果是我们手动加入的Cast算子，可分析其必要性，如果对精度没有影响，可去掉冗余的Cast、TransData算子。

如果是MindSpore自动生成的转换算子过多，可能是MindSpore框架针对某些特殊情况没有充分优化，可联系 [MindSpore社区](https://gitee.com/mindspore/mindspore/issues) 反馈。

#### 其他通用优化方法

- 使用自动混合精度

    混合精度训练方法是通过混合使用单精度和半精度数据格式来加速深度神经网络训练的过程，同时保持了单精度训练所能达到的网络精度。混合精度训练能够加速计算过程，同时减少内存使用和存取，并使得在特定的硬件上可以训练更大的模型或batch size。

    具体可参考[混合精度教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/enable_mixed_precision.html)。

- 使能图算融合

    图算融合是MindSpore特有的网络性能优化技术。它可以通过自动分析和优化现有网络计算图逻辑，并结合目标硬件能力，对计算图进行计算化简和替代、算子拆分和融合、算子特例化编译等优化，以提升设备计算资源利用率，实现对网络性能的整体优化。相比传统优化技术，图算融合具有多算子跨边界联合优化、与算子编译跨层协同、基于Polyhedral的算子即时编译等独特优势。另外，图算融合只需要用户打开对应配置后，整个优化过程即可自动完成，不需要网络开发人员进行其它额外感知，使得用户可以聚焦网络算法实现。

    图算融合的适用场景包括：对网络执行时间具有较高性能要求的场景；通过拼接基本算子实现自定义组合算子，并希望对这些基本算子进行自动融合，以提升自定义组合算子性能的场景。

    具体可参考[图片融合教程](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/enable_graph_kernel_fusion.html)。
