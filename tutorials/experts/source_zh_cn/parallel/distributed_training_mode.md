# 分布式并行训练模式

`Ascend` `GPU` `分布式并行`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/distributed_training_mode.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

目前MindSpore支持下述的四种并行模式：

- 数据并行：用户的网络参数规模在单卡上可以计算的情况下使用。这种模式会在每卡上复制相同的网络参数，训练时输入不同的训练数据，适合大部分用户使用。
- 半自动并行：用户的神经网络在单卡上无法计算，并且对切分的性能存在较大的需求。用户可以设置这种运行模式，手动指定每个算子的切分策略，达到较佳的训练性能。
- 自动并行：用户的神经网络在单卡上无法计算，但是不知道如何配置算子策略。用户启动这种模式，MindSpore会自动针对每个算子进行配置策略，适合想要并行训练但是不知道如何配置策略的用户。
- 混合并行：完全由用户自己设计并行训练的逻辑和实现，用户可以自己在网络中定义`AllGather`等通信算子。适合熟悉并行训练的用户。

在下面的文档中将会详细介绍这四种模式的用法和注意事项。

## 并行模式

当前MindSpore提供分布式并行训练的功能。它支持了上述的多种模式，可以通过`context.set_auto_parallel_context()`接口设置对应的并行模式。

在用户调用分布式训练流程时，需要调用如下代码进行通信的初始化，并且配置对应的rank_table_file，可以参考[分布式训练(Ascend)](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/distributed_training_ascend.html#多机多卡训练)的**多机多卡训练**章节。

```python
from mindspore.communication import init, get_rank, get_group_size
from mindspore import context
init()
device_num = get_group_size()
rank = get_rank()
print("rank_id is {}, device_num is {}".format(rank, device_num))
context.reset_auto_parallel_context()
# 下述的并行配置用户只需要配置其中一种模式
# 数据并行模式
context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL)
# 半自动并行模式
# context.set_auto_parallel_context(parallel_mode=context.ParallelMode.SEMI_AUTO_PARALLEL)
# 自动并行模式
# context.set_auto_parallel_context(parallel_mode=context.ParallelMode.AUTO_PARALLEL)
# 混合并行模式
# context.set_auto_parallel_context(parallel_mode=context.ParallelMode.HYBRID_PARALLEL)
```

下述涉及的自动并行接口，例如`context.set_auto_parallel_context`中的接口配置，可以在[分布式并行接口](https://mindspore.cn/docs/programming_guide/zh-CN/master/auto_parallel.html)进行查看。分布式并行训练在各场景的支持情况如下表。

| 并行模式 | 配置 | 动态图 | 静态图 | 支持设备 |
| ---------- | ------ | ------ | ---------- | ---------- |
| 数据并行   | DATA_PARALLEL | 支持   | 支持   | GPU、Ascend 910 |
| 半自动并行 | SEMI_AUTO_PARALLEL | 不支持 | 支持   | GPU、Ascend 910 |
| 全自动并行 | AUTO_PARALLEL | 不支持 | 支持   | GPU、Ascend 910 |
| 混合并行   | HYBRID_PARALLEL | 不支持 | 支持   | GPU、Ascend 910 |

### 数据并行

在数据并行中，用户定义网络的方式和单机脚本一样，但是在网络定义之前调用[init()](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.communication.html?highlight=init#mindspore.communication.init)去初始化设备通信状态。

```python
import numpy as np
from mindspore import Tensor, context, Model, Parameter
from mindspore.communication import init
from mindspore import ops, nn

class DataParallelNet(nn.Cell):
    def __init__(self):
        super(DataParallelNet, self).__init__()
        # 初始化权重
        weight_init = np.random.rand(512, 128).astype(np.float32)
        self.weight = Parameter(Tensor(weight_init))
        self.fc = ops.MatMul()
        self.reduce = ops.ReduceSum()

    def construct(self, x):
        x = self.fc(x, self.weight)
        x = self.reduce(x, -1)
        return x

init()
# 设置并行模式为数据并行，其他方式一致
context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL)
net = DataParallelNet()
model = Model(net)
model.train(*args, **kwargs)
```

### 半自动并行

相较于自动并行，半自动并行模式需要用户对算子手动配置切分**策略**实现并行。关于算子并行策略的定义可以参考这篇[设计文档](https://www.mindspore.cn/docs/zh-CN/master/design/distributed_training_design.html#自动并行原理)。

- 启动半自动和自动模式进行训练时，**必须**通过`model.train(*args, **kwargs)`接口进行训练，不支持自定义循环进行网络训练。

  ```python
  # 训练方式一：通过Model接口调用，仅支持这种方式
  model = Model(net, *args, **kwargs)
  model.train(*args, **kwargs)

  # 训练方式二：自定义循环，这种方式不支持
  for d in dataset.create_dict_iterator():
      for i in range(300):
          train_net(d["data"], d["label"])
          print(net_with_criterion(d["data"], d["label"]))
  ```

- 半自动并行模式相较于自动并行模式，需要用户**手动配置**每个算子的**shard**接口对并行策略进行调优。

    以`SemiAutoParallelNet`为例，在半自动并行模式下的脚本代码如下，`MatMul`的切分策略为`((1, 1),(1, 2))`，指定`self.weight`在第二维度上被切分两份。

    ```python
    import numpy as np
    from mindspore import Tensor, context, Model, Parameter
    from mindspore.communication import init
    from mindspore import ops, nn

    class SemiAutoParallelNet(nn.Cell):
        def __init__(self):
            super(SemiAutoParallelNet, self).__init__()
            # 初始化权重
            weight_init = np.random.rand(128, 128).astype(np.float32)
            self.weight = Parameter(Tensor(weight_init))
            self.weight2 = Parameter(Tensor(weight_init))
            # 设置切分策略。在construct中fc的输入有两个，第一个输入是x，第二个输入是权重self.weight
            # 因此shard需要提供一个tuple元组，分别对应每个输入tensor在对应维度的切分份数
            # (1,1)表示输入x的每一维度都没有切分
            # (1,2)表示在self.weight的第二维度上切成了两份
            # 切分的过程是在图编译的过程中，在编译完成后，self.weight的shape就会发生改变
            self.fc = ops.MatMul().shard(((1, 1),(1, 2)))
            self.reduce = ops.ReduceSum()

        def construct(self, x):
            x = self.fc(x, self.weight)
            # 在construct函数中去初始化并行调用operation算子时，相当于用户没有设置matmul算子的策略。那么默认的策略会自动配置数据并行，即((8, 1), (1, 1))。其中8表示用户此次运行的卡数
            x = ops.MatMul()(x, self.weight2)
            x = self.reduce(x, -1)
            return x

    init()
    context.set_auto_parallel_context(parallel_mode=context.ParallelMode.SEMI_AUTO_PARALLEL)
    net = SemiAutoParallelNet()
    model = Model(net)
    model.train(*args, **kwargs)
    ```

在前后算子的设备矩阵不一致时，会自动插入[重排布](https://www.mindspore.cn/docs/zh-CN/master/design/distributed_training_design.html?highlight=%E9%87%8D%E6%8E%92%E5%B8%83#id4), 确保`tensor`的切分状态符合下一个算子输入要求。例如在单机八卡的训练中，有下述的示例代码：

```python
import numpy as np
from mindspore import Tensor, Parameter
from mindspore import ops, nn
class SemiAutoParallelNet(nn.Cell):
    def __init__(self):
        super(SemiAutoParallelNet, self).__init__()
        # 初始化权重
        weight_init = np.random.rand(128, 128).astype(np.float32)
        self.weight = Parameter(Tensor(weight_init))
        self.weight2 = Parameter(Tensor(weight_init))
        # 设置切分策略
        self.fc = ops.MatMul().shard(((1, 1),(1, 2)))
        self.fc2 = ops.MatMul().shard(((8, 1),(1, 1)))
        self.reduce = ops.ReduceSum()

    def construct(self, x):
        # 在__init__中我们对fc的第二个输入配置为(1,2)
        # 所以经过fc的输出的tensor在输出第二维度上被切成了两份，从128变成了64，所以它的输出shape为 [batch, 64]，
        x = self.fc(x, self.weight)
        # 在__init__中我们通过shard的方式，对fc2第0个输入配置了(8,1)，表示要求此输入的第0维度切分成了8份
        # 而上一个算子fc的输出还是[batch,64]，第0维度并没有发生切分，因此存在一个tensor shape不一致的问题
        # 所以自动并行框架会在此处插入用户脚本中没有声明的StrideSlice算子，将x进行取切片操作
        # 以保证前后tensor shape的一致性。
        # 另外，fc的输出第1维度切分成了2份，但是fc2第0个输入的第1维度切分成了1份，因此还会插入allgather算子
        x = self.fc2(x, self.weight2)
        # 框架在此自动会插入一个AllGather算子和StridedSlice操作
        x = self.reduce(x, -1)
        return x
```

因此，如果前后的算子对输入的切分要求不一样，插入的重排布算子可能会有`AllGather`、`Split`、`Concat`和`StridedSlice`等算子。因此会增加网络的计算和通信耗时。用户可以[保存ir图](https://www.mindspore.cn/docs/zh-CN/master/design/mindir.html)查看整张网络的算子状态。其中自动并行流程产生的`ir`图名为`step_parallel_begin_xxxx.ir`和`step_parallel_end_xxxx.ir`。前者表示在进入并行流程之前图状态，后者表示经过自动并行流程处理后的图状态，用户可以查看后者这个文件，查找自动并行插入的算子。

> - 半自动并行模式时，未配置策略的算子默认以数据并行方式执行，对应的数据并行度为所有卡。
> - 自动并行模式支持通过策略搜索算法自动获取高效的算子并行策略，同时也支持用户对算子手动配置特定的并行策略。
> - 如果某个`parameter`被多个算子使用，则每个算子对这个`parameter`的切分策略需要保持一致，否则将报错。

自动和半自动模式中还可以通过对`Cell`配置`pipeline_stage`属性进行流水线并行，对应的流水线并行教程可以参考[应用流水线并行](https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/pipeline_parallel.html)。

### 全自动并行

自动并行模式，融合了数据并行、模型并行及混合并行的分布式并行模式，可以自动建立代价模型，找到训练时间较短的并行策略，为用户选择合适的并行模式。MindSpore提供了如下的三种不同的策略搜索算法：

- `dynamic_programming`：动态规划策略搜索算法。能够搜索出代价模型刻画的最优策略，但在搜索巨大网络模型的并行策略时耗时较长。其代价模型是围绕Ascend 910芯片基于内存的计算开销和通信开销对训练时间建模。
- `recursive_programming`：双递归策略搜索算法。对于巨大网络以及大规模多卡切分能够保证瞬间生成最优策略。其基于符号运算的代价模型可以自由适配不同的加速器集群。
- `sharding_propagation`：切分策略传播算法。由配置并行策略的算子向未配置的算子传播并行策略。在传播时，算法会尽量选取引发张量重排布通信最少的策略。关于算子的并行策略配置和张量重排布，可参考[半自动并行](#半自动并行)。

用户可以通过如下代码去设置上述的策略搜索算法：

```python
from mindspore import context
# 设置动态规划算法进行策略搜索
context.set_auto_parallel_context(parallel_mode=context.ParallelMode.AUTO_PARALLEL, search_mode="dynamic_programming")
# 设置双递归方法进行策略搜索
context.set_auto_parallel_context(parallel_mode=context.ParallelMode.AUTO_PARALLEL, search_mode="recursive_programming")
# 设置切分策略传播算法
context.set_auto_parallel_context(parallel_mode=context.ParallelMode.AUTO_PARALLEL, search_mode="sharding_propagation")
```

> - 在`sharding_propagation`模式下，算法根据用户设置的`shard`策略传播到整个模型，在`dynamic_programming`模式下，用户设置的`shard`策略也会生效，不会被搜索出来的策略覆盖掉。
> - 在全自动并行模式下，如果需要对某个Cell里的所有算子手动配置数据并行策略，可用Cell.set_data_parallel()统一设置。

### 混合并行

在MindSpore中特指用户通过手动切分模型实现混合并行的场景，用户可以在网络结构中定义通信算子原语`AllReduce`和`AllGather`等，手动执行并行流程。此时，用户需要自己实现参数的切分，算子切分后的通信等操作。例如下面的代码示例：

```python
import numpy as np
from mindspore import Tensor, context, Model
from mindspore.communication import init
from mindspore import ops, nn, Parameter

class HybridParallelNet(nn.Cell):
    def __init__(self):
        super(HybridParallelNet, self).__init__()
        # 以下2卡运行的场景为例子，实现分布式矩阵乘法来模拟单卡矩阵乘的结果。
        # 即原始的逻辑
        #        输入x,weight的shape分别为(32, 512), (512, 128)
        #        经过计算：matmul(x, weight)
        #        输出结果shape为(32, 128)的tensor
        # 下面我们手动实现上面的矩阵乘法逻辑
        # 我们需要手动的指定当前权重的切片的shape,我们希望在matmul的相关维度进行切分。相关维度切分的情况下
        # 需要对matmul的结果进行AllReduce操作，确保数值和单机的保持一致
        #
        # 分布式逻辑
        #         输入x,weight的shape分别为(32, 256), (256, 128)
        #         经过计算  output = matmul(x, weight)
        #                  output = allreduce(output)
        #         输出结果shape为(32, 128)的tensor
        weight_init = np.random.rand(256, 128).astype(np.float32)
        self.weight = Parameter(Tensor(weight_init))
        self.fc = ops.MatMul()
        self.reduce = ops.AllReduce()

    def construct(self, x):
        x = self.fc(x, self.weight)
        x = self.reduce(x)
        return x

init()
context.set_auto_parallel_context(parallel_mode=context.ParallelMode.HYBRID_PARALLEL)
net = HybridParallelNet()
model = Model(net)
model.train(*args, **kwargs)
```

## 数据导入方式

在并行训练中，支持三种数据的导入方式：

- 全量导入。仅在**半自动**和**全自动**并行模式下生效。用户可以通过`context.set_auto_parallel_context(full_batch=True)`开启。开启全量导入之后，在自动并行流程中认为读入的`batch`是一个网络输入的完整shape。例如，在8卡训练的情况下，假设每张卡`dataset`返回的shape是`[32, 8]`，那么当前一个迭代训练的训练的数据即为`[32, 8]`。因此，**用户需要保证每卡在每轮迭代输入的数据是一致的**。例如，确保每卡数据集的`shuffle`的顺序是一致的。

- 数据并行导入。用户不设置 `full_batch`的情况下，每卡读入的数据是当前训练迭代的一个分片，因此要求每卡读入的数据内容**不一样**。例如8卡训练的情况下，每卡读入数据的`shape`为`[32,8]`，那么当前一个迭代训练的数据总量为`[32*8, 8]`。

- 模型并行导入。模型并行导入的方式主要针对图像领域中图像尺寸太大无法在单卡进行计算时，直接在输入流程上就对图像进行切分。MindSpore在`context.set_auto_parallel_context`中提供了`dataset_strategy`接口，用户可以通过这个接口配置更加灵活的输入策略。注意，当用户使用此接口时，需要确保`dataset`返回的`tensor`符合对应的切分策略。如下代码所示：

  ```python
  from mindspore import context
  # 设置输入在第1维度上进行切分， 此时要求用户确保dataset返回的输入在第1维度上进行切分
  context.set_auto_parallel_context(dataset_strategy=((1, 8), (1, 8)))
  # 相当于设置full_batch=False
  context.set_auto_parallel_context(dataset_strategy="data_parallel")
  # 相当于设置full_batch=True
  context.set_auto_parallel_context(dataset_strategy="full_batch")
  ```

因此，在用户设置上述的配置之后，需要**手动**设置dataset的获取顺序，确保每卡的数据是期望的。
