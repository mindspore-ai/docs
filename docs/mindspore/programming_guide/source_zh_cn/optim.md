
# 优化器

`Linux` `Ascend` `GPU` `CPU` `模型开发`

<!-- TOC -->

- [优化器](#优化器)
    - [概述](#概述)
    - [权重配置](#权重配置)
        - [使用Cell的网络权重获取函数](#使用Cell的网络权重获取函数)
        - [自定义筛选](#自定义筛选)
    - [学习率](#学习率)
        - [固定学习率](#固定学习率)
        - [动态学习率：预生成学习率列表](#动态学习率：预生成学习率列表)
        - [动态学习率：定义学习率计算图](#动态学习率：定义学习率计算图)
    - [权重衰减](#权重衰减)
    - [参数分组](#参数分组)
    - [混合精度](#混合精度)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/optim.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

优化器在模型训练过程中，用于计算和更新网络参数，合适的优化器可以有效减少训练时间，提高最终模型性能。最基本的优化器是梯度下降（SGD），在此基础上，很多其他的优化器进行了改进，以实现目标函数能更快速更有效地收敛到全局最优点。

`mindspore.nn.optim`是MindSpore框架中实现各种优化算法的模块，包含常用的优化器、学习率等，接口具备较好的通用性，可以将以后更新、更复杂的方法集成到模块里。`mindspore.nn.optim`为模型提供常用的优化器，如`mindspore.nn.SGD`、`mindspore.nn.Adam`、`mindspore.nn.Ftrl`、`mindspore.nn.LazyAdam`、`mindspore.nn.Momentum`、`mindspore.nn.RMSProp`、`mindspore.nn.LARS`、`mindspore.nn.ProximalAadagrad`和`mindspore.nn.Lamb`等。同时`mindspore.nn`提供了动态学习率的模块，分为`dynamic_lr`和`learning_rate_schedule`，学习率的灵活设置可以有效支撑目标函数的收敛和模型的训练。

使用`mindspore.nn.optim`时，我们需要构建一个Optimizer实例。这个实例能够保持当前参数状态并基于计算得到的梯度进行参数更新。为了构建一个Optimizer，要指定需要优化的网络权重（必须是Parameter实例）的iterable，然后设置Optimizer的参数选项，比如学习率，权重衰减等。

以下内容分别从权重学习率、权重衰减、参数分组、混合精度等方面的配置分别进行详细介绍。

## 权重配置

在构建Optimizer实例时，通过`params`配置模型网络中要训练和更新的权重。`params`必须配置，常见的配置方法有以下两种。

### 使用Cell的网络权重获取函数

`Parameter`类中包含了一个`requires_grad`的布尔型的类属性，表征了模型网络中的权重是否需要梯度来进行更新（详情可参考：<https://gitee.com/mindspore/mindspore/blob/master/mindspore/common/parameter.py> ）。其中大部分权重的`requires_grad`的默认值都为True；少数默认为False，例如BatchNormalize中的`moving_mean`和`moving_variance`。用户可以根据需要，自行对`requires_grad`的值进行修改。

MindSpore提供了`get_parameters`方法来获取模型网络中所有权重，该方法返回了`Parameter`类型的网络权重；`trainable_params`方法本质是一个filter，过滤了`requires grad=True`的`Parameter`。用户在构建优化器时，可以通过配置`params`为`net.trainable_params()`来指定需要优化和更新的权重。

代码样例如下：

```python
import numpy as np
import mindspore.ops as ops
from mindspore import nn, Model, Tensor,  Parameter

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.conv = nn.Conv2d(1, 6, 5, pad_mode="valid")
        self.param = Parameter(Tensor(np.array([1.0], np.float32)))

    def construct(self, x):
        x = self.conv(x)
        x = x * self.param
        out = self.matmul(x, x)
        return out

net = Net()
optim = nn.Adam(params=net.trainable_params())
```

### 自定义筛选

用户也可以设定筛选条件，在使用`get_parameters`获取到网络全部参数后，通过限定参数名字等方法，自定义filter来决定哪些参数需要更新。例如下面的例子，训练过程中将只对非卷积参数进行更新：

```python
from mindspore import nn

params_all = net.get_parameters()
no_conv_params = list(filter(lambda x: 'conv' not in x.name, params_all))
optim = nn.Adam(params=no_conv_params, learning_rate=0.1, weight_decay=0.0)
```

## 学习率

学习率作为机器学习及深度学习中常见的超参，对目标函数能否收敛到局部最小值及何时收敛到最小值有重要作用。学习率过大容易导致目标函数波动较大，难以收敛到最优值，太小则会导致收敛过程耗时长，除了基本的固定值设置，很多动态学习率的设置方法也在深度网络的训练中取得了显著的效果。

### 固定学习率

使用固定学习率时，优化器传入的`learning_rate`为浮点类型或标量Tensor。

以Momentum为例，固定学习率为0.01，用法如下：

```python
from mindspore import nn

net = Net()
optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.01, momentum=0.9)
loss = nn.SoftmaxCrossEntropyWithLogits()
model = Model(net, loss_fn=loss, optimizer=optim)
```

### 动态学习率

模块提供了动态学习率的两种不同的实现方式，`dynamic_lr`和`learning_rate_schedule`：

- `dynamic_lr`: 预生成长度为`total_step`的学习率列表，将列表传入优化器中使用， 训练过程中， 第i步使用第i个学习率的值作为当前step的学习率，其中，`total_step`的设置值不能小于训练的总步数；

- `learning_rate_schedule`: 优化器学习率指定一个LearningRateSchedule的Cell实例，学习率会和训练网络一起组成计算图，在执行过程中，根据step计算出当前学习率。

#### 预生成学习率列表

`mindspore.nn.dynamic_lr`模块有以下几个类，分别使用不同的数学计算方法对学习率进行计算：

- `piecewise_constant_lr`类：基于得到分段不变的学习速率。

- `exponential_decay_lr`类：基于指数衰减函数计算学习率。

- `natural_exp_decay_lr`类：基于自然指数衰减函数计算学习率。

- `inverse_decay_lr`类：基于反时间衰减函数计算学习速率。

- `cosine_decay_lr`类：基于余弦衰减函数计算学习率。

- `polynomial_decay_lr`类：基于多项式衰减函数计算学习率。

- `warmup_lr`类：提高学习率。

它们属于`dynamic_lr`的不同实现方式。

以`piecewise_constant_lr`为例：

```python
from mindspore import nn

milestone = [2, 5, 10]
learning_rates = [0.1, 0.05, 0.01]
lr = nn.dynamic_lr.piecewise_constant_lr(milestone, learning_rates)
print(lr)
```

输出结果如下：

```text
[0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
```

#### 定义学习率计算图

`mindspore.nn.learning_rate_schedule`模块下有以下几个：`ExponentialDecayLR`类、`NaturalExpDecayLR`类、`InverseDecayLR`类、`CosineDecayLR`类、`PolynomialDecayLR`类和`WarmUpLR`类。它们都属于`learning_rate_schedule`，只是实现方式不同，各自含义如下：

- `ExponentialDecayLR`类：基于指数衰减函数计算学习率。

- `NaturalExpDecayLR`类：基于自然指数衰减函数计算学习率。

- `InverseDecayLR`类：基于反时间衰减函数计算学习速率。

- `CosineDecayLR`类：基于余弦衰减函数计算学习率。

- `PolynomialDecayLR`类：基于多项式衰减函数计算学习率。

- `WarmUpLR`类：提高学习率。

它们属于`learning_rate_schedule`的不同实现方式。

例如`ExponentialDecayLR`类代码样例如下：

```python
from mindspore import nn
from mindspore import Tensor
from mindspore import dtype as mstype

polynomial_decay_lr = nn.learning_rate_schedule.PolynomialDecayLR(learning_rate=0.1,
                                   end_learning_rate=0.01,
                                   decay_steps=4,
                                   power=0.5 )

global_step = Tensor(2, mstype.int32)
result = polynomial_decay_lr(global_step)
print(result)
```

输出结果如下：

```text
0.0736396
```

## 权重衰减

一般情况下，weight_decay取值范围为[0, 1)，实现对（BatchNorm以外的）参数使用权重衰减的策略，以避免模型过拟合问题；weight_decay的默认值为0.0，此时不使用权重衰减策略。

```python
net = Net()
optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9, weight_decay=0.9)
```

## 参数分组

优化器也支持为不同参数单独设置选项，此时不直接传入变量，而是传入一个字典的列表，每个字典定义一个参数组别的设置值，key可以为“params”，“lr”，“weight_decay”，”grad_centralizaiton”，value为对应的设定值。`params`必须配置，其余参数可以选择配置，未配置的参数项，将采用定义优化器时设置的参数值。

分组时，学习率可以使用固定学习率，也可以使用dynamic_lr和learningrate_schedule动态学习率。

> 当前MindSpore除个别优化器外（例如AdaFactor，FTRL），均支持对学习率进行分组，详情参考各优化器的说明。

例如下面的例子:

- conv_params组别的参数，使用固定学习率0.01， `weight_decay`为字典传入的数值0.01；

- no_conv_params组别使用`learning_rate_schedule`的动态学习率`PolynomialDecayLR`， `weight_decay`使用优化器配置的值0.0；

- group_params还提供了`order_params`配置项；一般情况下`order_params`无需配置。group_params根据分组情况会改变parameter和梯度的计算顺序。如果使用自动并行策略，并通过`set_all_reduce_fusion_split_indices`配置了梯度更新的切分点，group_params引起的顺序变化会影响梯度广播的并行效果，此时可以通过`order_params`指定预期的参数顺序，例如指定为`net.trainable_params()`，使参数顺序与网络中定义权重的原始顺序保持一致。

```python
from mindspore import nn

net = Net()

# Use parameter groups and set different values
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))

fix_lr = 0.01
polynomial_decay_lr = nn.learning_rate_schedule.PolynomialDecayLR(learning_rate=0.1,
                                   end_learning_rate=0.01,
                                   decay_steps=4,
                                   power=0.5 )

group_params = [{'params': conv_params, 'weight_decay': 0.01, 'lr': fix_lr},
                {'params': no_conv_params, 'lr': polynomial_decay_lr},
                {'order_params': net.trainable_params()}]

optim = nn.Momentum(group_params, learning_rate=0.1, momentum=0.9, weight_decay=0.0)
```

## 混合精度

深度神经网络存在使用混合精度训练的场景，这种方法通过混合使用单精度和半精度数据格式来加速网络训练，同时保持了单精度训练所能达到的网络精度。混合精度训练能够加速计算过程，减少内存使用和存取，并使得在特定的硬件上可以训练更大的模型或batch size。

在混合精度训练过程中，会使用float16类型来替代float32类型存储数据，但由于float16类型数据比float32类型数据范围小很多，所以当某些参数（例如梯度）在训练过程中变得很小时，就会发生数据下溢。为避免半精度float16类型数据下溢，MindSpore提供了`FixedLossScaleManager`和`DynamicLossScaleManager`方法。其主要思想是计算loss时，将loss扩大一定的倍数，由于链式法则的存在，梯度也会相应扩大，然后在优化器更新权重时再缩小相应的倍数，从而避免了数据下溢的情况又不影响计算结果。

一般情况下优化器不需要与`LossScale`功能配合使用，但使用`FixedLossScaleManager`，并且`drop_overflow_update`为False时，优化器需设置`loss_scale`的值，且`loss_scale`值与`FixedLossScaleManager`的相同，具体用法详见：<https://www.mindspore.cn/docs/programming_guide/zh-CN/master/lossscale.html>。
