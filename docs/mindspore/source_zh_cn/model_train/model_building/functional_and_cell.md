# Functional与Cell

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.1/docs/mindspore/source_zh_cn/model_train/model_building/functional_and_cell.md)

## 算子Functional接口

MindSpore框架提供了丰富的Functional接口，这些接口定义在 `mindspore.ops` 下，以函数的形式直接定义了操作或计算过程，无需显式创建算子类实例。Functional接口提供了包括神经网络层函数、数学运算函数、Tensor操作函数、Parameter操作函数、微分函数、调试函数等类型的接口，这些接口可以直接在 `Cell` 的 `construct` 方法中使用，也可以作为独立的操作在数据处理或模型训练中使用。

MindSpore在 `Cell` 里使用Functional接口的流程如下所示：

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class MyCell(nn.Cell):
    def construct(self, x, y):
        output = ops.add(x, y)
        return output

net = MyCell()
x = ms.Tensor([1, 2, 3], ms.float32)
y = ms.Tensor([4, 5, 6], ms.float32)
output = net(x, y)
print(output)
```

运行结果：

```text
[5. 7. 9.]
```

MindSpore独立使用Functional接口的流程如下所示：

```python
import mindspore as ms
import mindspore.ops as ops

x = ms.Tensor([1, 2, 3], ms.float32)
y = ms.Tensor([4, 5, 6], ms.float32)
output = ops.add(x, y)
print(output)
```

运行结果：

```text
[5. 7. 9.]
```

## 网络基本构成单元 Cell

MindSpore框架中的核心构成单元 `mindspore.nn.Cell` 是构建神经网络的基本模块，负责定义网络的计算逻辑。`Cell` 不仅支持动态图（PyNative模式）下作为网络的基础组件，也能够在静态图（GRAPH模式）下被编译成高效的计算图执行。`Cell` 通过其 `construct` 方法定义了前向传播的计算过程，并可通过继承机制扩展功能，实现自定义的网络层或复杂结构。通过 `set_train` 方法，`Cell` 能够灵活地在训练与推理模式间切换，以适应不同算子在两种模式下的行为差异。此外，`Cell` 还提供了丰富的API，如混合精度、参数管理、梯度设置、Hook功能、重计算等，以支持模型的优化与训练。

MindSpore 基本的 `Cell` 搭建过程如下所示：

```python
import mindspore.nn as nn
import mindspore.ops as ops

class MyCell(nn.Cell):
    def __init__(self, forward_net):
        super(MyCell, self).__init__(auto_prefix=True)
        self.net = forward_net
        self.relu = ops.ReLU()

    def construct(self, x):
        y = self.net(x)
        return self.relu(y)

inner_net = nn.Conv2d(120, 240, 4, has_bias=False)
my_net = MyCell(inner_net)
print(my_net.trainable_params())
```

运行结果：

```text
[Parameter (name=net.weight, shape=(240, 120, 4, 4), dtype=Float32, requires_grad=True)]
```

MindSpore中，参数的名字一般是根据`__init__`定义的对象名字和参数定义时用的名字组成的，比如上面的例子中，卷积的参数名为`net.weight`，其中，`net`是`self.net = forward_net`中的对象名，`weight`是Conv2d中定义卷积的参数时的`name`：`self.weight = Parameter(initializer(self.weight_init, shape), name='weight')`。

### Parameter管理

MindSpore 有两种数据对象：`Tensor` 和 `Parameter` 。其中 `Tensor` 对象仅参与运算，并不需要对其进行梯度求导和Parameter更新，而 `Parameter` 会根据其属性`requires_grad` 来决定是否传入优化器中。

#### Parameter获取

`mindspore.nn.Cell` 使用 `parameters_dict` 、`get_parameters` 和 `trainable_params` 接口获取 `Cell` 中的 `Parameter` 。

- parameters_dict：获取网络结构中所有Parameter，返回一个以key为Parameter名，value为Parameter值的`OrderedDict`。

- get_parameters：获取网络结构中的所有Parameter，返回`Cell`中`Parameter`的迭代器。

- trainable_params：获取`Parameter`中`requires_grad`为`True`的属性，返回可训练的Parameter的列表。

在定义优化器时，使用`net.trainable_params()`获取需要进行Parameter更新的Parameter列表。

```python
import mindspore.nn as nn

net = nn.Dense(2, 1, has_bias=True)
print(net.trainable_params())

for param in net.trainable_params():
    param_name = param.name
    if "bias" in param_name:
        param.requires_grad = False
print(net.trainable_params())
```

运行结果：

```text
[Parameter (name=weight, shape=(1, 2), dtype=Float32, requires_grad=True), Parameter (name=bias, shape=(1,), dtype=Float32, requires_grad=True)]
[Parameter (name=weight, shape=(1, 2), dtype=Float32, requires_grad=True)]
```

#### Parameter保存和加载

MindSpore提供了`load_checkpoint`和`save_checkpoint`方法用来Parameter的保存和加载，需要注意的是Parameter保存时，保存的是Parameter列表，Parameter加载时对象必须是Cell。
在Parameter加载时，可能Parameter名对不上，需要做一些修改，可以直接构造一个新的Parameter列表给到`load_checkpoint`加载到Cell。

```python
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn

net = nn.Dense(2, 1, has_bias=True)
for param in net.get_parameters():
    print(param.name, param.data.asnumpy())

ms.save_checkpoint(net, "dense.ckpt")
dense_params = ms.load_checkpoint("dense.ckpt")
print(dense_params)
new_params = {}
for param_name in dense_params:
    print(param_name, dense_params[param_name].data.asnumpy())
    new_params[param_name] = ms.Parameter(ops.ones_like(dense_params[param_name].data), name=param_name)

ms.load_param_into_net(net, new_params)
for param in net.get_parameters():
    print(param.name, param.data.asnumpy())
```

运行结果：

```text
weight [[-0.0042482  -0.00427286]]
bias [0.]
{'weight': Parameter (name=weight, shape=(1, 2), dtype=Float32, requires_grad=True), 'bias': Parameter (name=bias, shape=(1,), dtype=Float32, requires_grad=True)}
weight [[-0.0042482  -0.00427286]]
bias [0.]
weight [[1. 1.]]
bias [1.]
```

### 子模块管理

`mindspore.nn.Cell` 中可定义其他Cell实例作为子模块。这些子模块是网络中的组成部分，自身也可能包含可学习的Parameter（如卷积层的权重和偏置）和其他子模块。这种层次化的模块结构允许用户构建复杂且可重用的神经网络架构。

`mindspore.nn.Cell` 提供 `cells_and_names` 、 `insert_child_to_cell` 等接口实现子模块管理功能。

```python
from mindspore import nn

class MyCell(nn.Cell):
    def __init__(self):
        super(MyCell, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # 使用insert_child_to_cell添加子模块
        self.insert_child_to_cell('conv3', nn.Conv2d(64, 128, 3, 1))

        self.sequential_block = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1),
            nn.ReLU()
        )

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.sequential_block(x)
        return x

module = MyCell()

# 使用cells_and_names遍历所有子模块（包括直接和间接子模块）
for name, cell_instance in module.cells_and_names():
    print(f"Cell name: {name}, type: {type(cell_instance)}")
```

运行结果：

```text
Cell name: , type: <class '__main__.MyCell'>
Cell name: conv1, type: <class 'mindspore.nn.layer.conv.Conv2d'>
Cell name: conv2, type: <class 'mindspore.nn.layer.conv.Conv2d'>
Cell name: conv3, type: <class 'mindspore.nn.layer.conv.Conv2d'>
Cell name: sequential_block, type: <class 'mindspore.nn.layer.container.SequentialCell'>
Cell name: sequential_block.0, type: <class 'mindspore.nn.layer.activation.ReLU'>
Cell name: sequential_block.1, type: <class 'mindspore.nn.layer.conv.Conv2d'>
Cell name: sequential_block.2, type: <class 'mindspore.nn.layer.activation.ReLU'>
```

### Hook功能

调试深度学习网络是每一个深度学习领域的从业者需要面对且投入精力较大的工作。由于深度学习网络隐藏了中间层算子的输入、输出数据以及反向梯度，只提供网络输入数据（特征量、权重）的梯度，导致无法准确地感知中间层算子的数据变化，从而降低了调试效率。为了方便用户准确、快速地对深度学习网络进行调试，MindSpore在动态图模式下设计了Hook功能，使用Hook功能可以捕获中间层算子的输入、输出数据以及反向梯度。

目前，动态图模式下 `MindSpore.nn.Cell` 提供了四种形式的Hook功能，分别是：`register_forward_pre_hook` 、 `register_forward_hook` 、`register_backward_hook` 和 `register_backward_pre_hook` 功能。详见[Hook编程](https://www.mindspore.cn/docs/zh-CN/r2.4.1/model_train/custom_program/hook_program.html)。

### 重计算

MindSpore采用反向模式的自动微分，根据正向图计算流程来自动推导出反向图，正向图和反向图一起构成了完整的计算图。在计算某些反向算子时，需要用到一些正向算子的计算结果，导致这些正向算子的计算结果需要驻留在内存中，直到依赖它们的反向算子计算完，这些正向算子的计算结果占用的内存才会被复用。这一现象推高了训练的内存峰值，在大规模网络模型中尤为显著。

为了解决这个问题，`mindspore.nn.Cell.recompute` 接口提供了重计算的功能。重计算功能可以不保存正向算子的计算结果，让这些内存可以被复用，在计算反向算子时，如果需要正向的结果，再重新计算正向算子。详见[重计算](https://www.mindspore.cn/docs/zh-CN/r2.4.1/model_train/parallel/recompute.html)。
