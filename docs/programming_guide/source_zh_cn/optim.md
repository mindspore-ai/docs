# 优化算法

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/programming_guide/source_zh_cn/optim.md)
&nbsp;&nbsp;
[![查看notebook](./_static/logo_notebook.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r1.1/programming_guide/mindspore_optim.ipynb)
&nbsp;&nbsp;
[![在线运行](./_static/logo_modelarts.png)](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/notebook/loading?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL3Byb2dyYW1taW5nX2d1aWRlL21pbmRzcG9yZV9vcHRpbS5pcHluYg==&image_id=65f636a0-56cf-49df-b941-7d2a07ba8c8c)

## 概述

`mindspore.nn.optim`是MindSpore框架中实现各种优化算法的模块，包含常用的优化器、学习率等，并且接口具备足够的通用性，可以将以后更新、更复杂的方法集成到模块里。

`mindspore.nn.optim`为模型提供常用的优化器，如`SGD`、`ADAM`、`Momentum`。优化器用于计算和更新梯度，模型优化算法的选择直接关系到最终模型的性能，如果有时候效果不好，未必是特征或者模型设计的问题，很有可能是优化算法的问题；同时还有`mindspore.nn`提供的学习率的模块，学习率分为`dynamic_lr`和`learning_rate_schedule`，都是动态学习率，但是实现方式不同，学习率是监督学习以及深度学习中最为重要的参数，其决定着目标函数是否能收敛到局部最小值以及何时能收敛到最小值。合适的学习率能够使目标函数在合适的时间内收敛到局部最小值。

> 本文档中的所有示例，支持CPU，GPU，Ascend环境。

## 学习率

### dynamic_lr

`mindspore.nn.dynamic_lr`模块有以下几个类：

- `piecewise_constant_lr`类：基于得到分段不变的学习速率。
- `exponential_decay_lr`类：基于指数衰减函数计算学习率。
- `natural_exp_decay_lr`类：基于自然指数衰减函数计算学习率。
- `inverse_decay_lr`类：基于反时间衰减函数计算学习速率。
- `cosine_decay_lr`类：基于余弦衰减函数计算学习率。
- `polynomial_decay_lr`类：基于多项式衰减函数计算学习率。
- `warmup_lr`类：提高学习率。

它们是属于`dynamic_lr`的不同实现方式。

例如`piecewise_constant_lr`类代码样例如下：

```python
from mindspore.nn.dynamic_lr import piecewise_constant_lr

def test_dynamic_lr():
    milestone = [2, 5, 10]
    learning_rates = [0.1, 0.05, 0.01]
    lr = piecewise_constant_lr(milestone, learning_rates)
    print(lr)


if __name__ == '__main__':
    test_dynamic_lr()
```

返回结果如下：

```text
[0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
```

### learning_rate_schedule

`mindspore.nn.learning_rate_schedule`模块下有以下几个类：`ExponentialDecayLR`类、`NaturalExpDecayLR`类、`InverseDecayLR`类、`CosineDecayLR`类、`PolynomialDecayLR`类和`WarmUpLR`类。它们都属于`learning_rate_schedule`，只是实现方式不同，各自含义如下：

- `ExponentialDecayLR`类：基于指数衰减函数计算学习率。
- `NaturalExpDecayLR`类：基于自然指数衰减函数计算学习率。
- `InverseDecayLR`类：基于反时间衰减函数计算学习速率。
- `CosineDecayLR`类：基于余弦衰减函数计算学习率。
- `PolynomialDecayLR`类：基于多项式衰减函数计算学习率。
- `WarmUpLR`类：提高学习率。

它们是属于`learning_rate_schedule`的不同实现方式。

例如ExponentialDecayLR类代码样例如下：

```python
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore.nn.learning_rate_schedule import ExponentialDecayLR

def test_learning_rate_schedule():
    learning_rate = 0.1    # learning_rate(float) - The initial value of learning rate.
    decay_rate = 0.9    # decay_rate(float) - The decay rate.
    decay_steps = 4    # decay_steps(int) - A value used to calculate decayed learning rate.
    global_step = Tensor(2, mstype.int32)
    exponential_decay_lr = ExponentialDecayLR(learning_rate, decay_rate, decay_steps)
    res = exponential_decay_lr(global_step)
    print(res)


if __name__ == '__main__':
    test_learning_rate_schedule()
```

返回结果如下：

```text
0.094868325
```

## Optimzer

### 如何使用

为了使用`mindspore.nn.optim`，我们需要构建一个`Optimizer`对象。这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。

- 构建

为了构建一个`Optimizer`，我们需要给它一个包含可需要优化的参数（必须是Variable对象）的iterable。然后，你可以设置Optimizer的参数选项，比如学习率，权重衰减等等。

代码样例如下：

```python
from mindspore import nn

optim = nn.SGD(group_params, learning_rate=0.1, weight_decay=0.0)
optim = nn.Adam(params=net.trainable_params())

optim = nn.Adam(group_params, learning_rate=0.1, weight_decay=0.0)

```

- 为每一个参数单独设置选项

优化器也支持为每个参数单独设置选项。若想这么做，不要直接传入变量Variable，而是传入一个字典的iterable。每一个字典都分别定义了一组参数，并且包含一个key键，这个key键对应相应的参数value值。其他的key键应该是优化器所接受的其他参数，并且会被用于对这组参数的优化。

我们仍然能够传递选项作为关键字参数，在未重写这些选项的组中，它们会被用作默认值。当你只想改动一个参数组的选项，但其他参数组的选项不变时，这是非常有用的。
例如，当我们想制定每一层的学习率时，以`SGD`为例：

```python
from mindspore import nn

optim = nn.SGD([{'params': conv_params, 'weight_decay': 0.01},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}],
               learning_rate=0.1, weight_decay=0.0)

```

这段示例意味着当参数是conv_params时候，权重衰减使用的是0.01，学习率使用的是0.1；而参数是no_conv_params时候，权重衰减使用的是0.0，学习率使用的是0.01。这个学习率learning_rate=0.1会被用于所有分组里没有设置学习率的参数，权重衰减weight_deca也是如此。

### 内置优化器

深度学习优化算法大概常用的有`SGD`、`Adam`、`Ftrl`、`lazyadam`、`Momentum`、`RMSprop`、`Lars`、`Proximal_ada_grad`和`lamb`这几种。
在`mindspore.nn.optim`模块中，他们都有对应的类实现。例如：

- `SGD`，默认参数为纯SGD，设置`momentum`参数不为0，考虑了一阶动量，设置`nesterov`为True后变成`NAG`，即`Nesterov Accelerated Gradient`，在计算梯度时计算的是向前走一步所在位置的梯度。

- `RMSprop`，考虑了二阶动量，对于不同的参数有不同的学习率，即自适应学习率，对`Adagrad`进行了优化，通过指数平滑只考虑一定窗口内的二阶动量。

- `Adam`，同时考虑了一阶动量和二阶动量，可以看成`RMSprop`上进一步考虑了一阶动量。

例如`SGD`的代码样例如下：

```python
from mindspore import nn, Tensor, Model
import mindspore.ops as ops
import numpy as np
from mindspore import dtype as mstype
from mindspore import Parameter

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.conv = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.z = Parameter(Tensor(np.array([1.0], np.float32)))

    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out

net = Net()
optim = nn.SGD(params=net.trainable_params())

conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
group_params = [{'params': conv_params, 'weight_decay': 0.01},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}]
optim = nn.SGD(group_params, learning_rate=0.1, weight_decay=0.0)

loss = nn.SoftmaxCrossEntropyWithLogits()
model = Model(net, loss_fn=loss, optimizer=optim)

```
