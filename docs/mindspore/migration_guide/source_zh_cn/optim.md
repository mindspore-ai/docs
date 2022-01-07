# 优化器迁移指南

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/optim.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

优化器在模型训练过程中，用于计算和更新网络参数，本文对比MindSpore和PyTorch的在这一部分的实现方式差异，分别从基本用法、基类入参设置及支持的方法、自定义优化器、API映射四部分展开。

## 基本用法

MindSpore：使用优化器时，通常需要预先定义网络、损失函数和优化器：

```python
from mindspore import context, Tensor, ParameterTuple
from mindspore import nn, Model, ops
import numpy as np
from mindspore import dtype as mstype

class Net(nn.Cell):
  def __init__(self):
    super(Net, self).__init__()
    self.conv = nn.Conv2d(3, 64, 3)
    self.bn = nn.BatchNorm2d(64)
  def construct(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return x

net = Net()
loss = nn.MSELoss()
optimizer = nn.SGD(params=net.trainable_params(), learning_rate=0.01)
```

在MindSpore中，定义好网络、损失函数、优化器后，一般在以下三种场景下使用：

- MindSpore封装了`Model`高阶API来方便用户定义和训练网络，在定义`Model`时指定优化器；

    ```python
    # 使用Model接口
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={"accuracy"})
    ```

- MindSpore提供了`TrainOneStepCell`接口，通过传入优化器和一个`WithLossCell`的实例，自定义训练网络；

    ```python
    # 使用TrainOneStepCell自定义网络
    loss_net = nn.WithLossCell(net, loss) # 包含损失函数的Cell
    train_net = nn.TrainOneStepCell(loss_net, optimizer)
    train_dataset = [(Tensor(np.random.rand(1, 3, 64, 32), mstype.float32), Tensor(np.random.rand(1, 64, 64, 32), mstype.float32))]
    for i in range(5):
        for image, label in train_dataset:
            train_net.set_train()
            res = train_net(image, label) # 执行网络的单步训练
    ```

- 在PyNative模式下，实现单步执行优化器。

    ```python
    # pynative模式下，单步实现GradOperation求梯度，并执行优化器
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    class GradWrap(nn.Cell):
      """ GradWrap definition """
      def __init__(self, network):
          super(GradWrap, self).__init__(auto_prefix=False)
          self.network = network
          self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

      def construct(self, x, label):
          weights = self.weights
          return ops.GradOperation(get_by_list=True)(self.network, weights)(x, label)

          loss_net = nn.WithLossCell(net, loss)
          train_network = GradWrap(loss_net)

          output = net(image)
          loss_output = loss(output, label)
          grads = train_network(image, label)
          success = optimizer(grads)
    ```

PyTorch：PyTorch为`Tensor`建立了`grad`属性和`backward`方法，`tensor.grad`是通过`tensor.backward`方法（本质是`PyTorch.autograd.backward`）计算的，且在计算中进行梯度值累加，因此一般在调用`tensor.backward`方法前，需要手动将`grad`属性清零。MindSpore没有为`Tensor`和`grad`建立直接联系，在使用时不需要手动清零。

在下面的代码中，初始化了一个优化器实例，每次循环调用`zero_grad`清零梯度，`backward`更新梯度，`step`更新网络参数，返回损失值。

```python
import torch
from torch import optim, nn
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
train_dataset = [(torch.tensor(np.random.rand(1, 3, 64, 32).astype(np.float32)), torch.tensor(np.random.rand(1, 64, 62, 30).astype(np.float32)))]

for epoch in range(5):
    for image, label in train_dataset:
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
```

## 基类入参设置及支持的方法

### 基类入参

MindSpore：

```python
optimizer(learning_rate, parameters, weight_decay=0.0, loss_scale=1.0)
```

PyTorch：

```python
optimizer(params, defaults)
```

#### 网络中需要被训练的参数

MindSpore和PyTorch的优化器都需要传入网络中需要被训练的参数，且参数的设置同时都支持默认接口和用户自定义设置两种方式。

- 默认接口：

    MindSpore的`parameter`包含了网络中所有的参数，通过`require_grad`属性来区分是否需要训练和优化。`trainable_params`方法返回一个`filter`的`list`，筛选了网络中`require_grad`属性为True的`parameter`。

    ```python
    from mindspore import nn
    optim_sgd = nn.SGD(net.trainable_params())
    ```

    PyTorch的`state`包含了网络中所有的参数，其中需要被优化的是`parameter`，不需要优化的是`buffer`（例如：BatchNorm中的`running_mean`和`running_var`     ）。`parameters`方法返回需要被优化参数的`generator`。

    ```python
    from torch import nn, optim
    optim_sgd = optim.SGD(params=model.parameters(), lr=0.01)
    ```

- 用户自定义：

    MindSpore和PyTorch都支持用户自定义传入需要优化的参数，例如，对非卷积参数进行训练和优化。代码样例如下：

    ```python
    from mindspore import nn

    net = Net()
    all_params = net.get_parameters()
    non_conv_params = list(filter(lambda x: "conv" not in x.name, all_params))
    optim_sgd = nn.SGD(params=non_conv_params)
    ```

    ```python
    from torch import optim

    net = Net()
    all_params = model.named_parameters()
    target_params = []
    for name, params in all_params:
        if "conv" in name:
            target_params.append(params)
    optim_sgd = optim.SGD(params=target_params, lr=0.01)
    ```

#### 学习率

使用固定学习率时，用法相同，传入固定值即可；使用动态学习率时，MindSpore和PyTorch都支持动态学习率调整策略，实现方式略有不同。

- MindSpore：动态学习率有两种实现方式，预生成列表`mindspore.nn.dynamic_lr`和计算图格式`mindspore.nn.learning_rate_schedule`，且动态学习率实例作为优化器的参数输入。以预生成学习率列表的`piecewise_constant_lr`为例：

    ```python
    from mindspore import nn

    milestone = [2, 5, 10]
    learning_rates = [0.1, 0.05, 0.01]
    lr = nn.dynamic_lr.piecewise_constant_lr(milestone, learning_rates)
    print(lr)
    ```

    ```text
    out: [0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
    ```

- PyTorch：优化器作为`lr_scheduler`的输入，调用`step`方法对学习率进行更新。

    ```python
    from torch import optim

    model = Net()
    optimizer = optim.SGD(model.parameters(), 0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(5):
        for input, target in train_dataset:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(scheduler.get_last_lr())

    # out:
    # [0.09000000000000001]
    # [0.08100000000000002]
    # [0.07290000000000002]
    # [0.06561000000000002]
    # [0.05904900000000002]
    ```

调整策略映射表

| mindspore.nn.dynamic_lr | mindspore.nn.learning_rate_schedule | PyTorch.optim.lr_scheduler |
|:--|:--|:--|
| `piecewise_constant_lr`：分段不变 | / |  `StepLR`: 每隔step_size个epoch，学习率乘gamma；`MultiStepLR`: epoch为milestones的时候学习率乘️gamma
|`exponential_decay_lr`：指数衰减| `ExponentialDecayLR`：指数衰减 | `ExponentialLR`: 指数衰减，lr = lr * (学习率乘gamma^epoch)
| `natural_exp_decay_lr`：自然指数衰减 | `NaturalExpDecayLR`：自然指数衰减 |  /
| `inverse_decay_lr`：反时间衰减 | `InverseDecayLR`：反时间衰减 |  /
| `cosine_decay_lr`：余弦衰减|`CosineDecayLR`：余弦衰减  |  `CosineAnnealingLR`: 余弦衰减
|`polynomial_decay_lr`：多项式衰减 | `PolynomialDecayLR`：多项式衰减 |  /
| /|/ | `CosineAnnealingWarmRestarts`：周期变化余弦衰减
| /|/ |   `CyclicLR/OneCycleLR`：三角循环
| /|/ |  `ReduceLROnPlateau`：自适应调整
| /|/ |   `LambdaLR`：传入Lambda函数，自定义调整
| /|/ |  `MultiplicativeLR`：乘上lr_lambda中设置的数值

#### weight decay

用法相同。一般情况下，weight_decay取值范围为\[0, 1\)，实现对需要优化的参数使用权重衰减的策略，以避免模型过拟合问题；weight_decay的默认值为0.0，此时不使用权重衰减策略。

#### 参数分组

MindSpore和PyTorch都支持参数分组且使用方法相似，在使用时都是给优化器传入一个字典的列表，每个字典对应一个参数组，其中key为参数名，value为对应的设置值。不同点是，MindSpore只支持对“lr”，“weight_decay”，“grad_centralizaiton”实现分组，pytoch支持对所有参数进行分组。此外，PyTorch还支持`add_param_group`方法，对参数组进行添加和管理。

> MindSpore和PyTorch各自有部分优化器不支持参数分组，请参考具体优化器的实现。

MindSpore参数分组用法请参考[编程指南](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/optim.html#id11)；PyTorch参数分组用法参考下述样例：

```python
from torch import optim

net = Net()
all_params = net.parameters()
conv_params = []
non_conv_params = []
# 根据自己的筛选规则 将所有网络参数进行分组
for pname, p in model.named_parameters():
    if ('conv' in pname):
        conv_params += [p]
    else:
        non_conv_params += [p]

print(len(conv_params), len(non_conv_params))
# 构建不同学习参数的优化器
optimizer = torch.optim.SGD([
        {'params': conv_params, 'lr': 0.02},
        {'params': non_conv_params, 'weight_decay': 0.5}],
        lr=0.01, momentum=0.9)

# out: 2 2
```

#### 混合精度

MindSpore中的混合精度场景下，如果使用`FixedLossScaleManager`进行溢出检测，且`drop_overflow_update`为False时，优化器需设置`loss_scale`的值，且`loss_scale`值与`FixedLossScaleManager`的相同，详细使用方法可以参考[优化器的混合精度配置](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/optim.html#id12)。PyTorch的混合精度设置不作为优化器入参。

### 基类支持的方法

#### 获取LR

`torch.optim.lr_scheduler.get_last_lr()`：根据参数组返回对应的最新学习率数值的列表。

mindspore中没有直接可以按照组别获取对应学习率的功能，但提供了以下方法辅助使用：

- `mindspore.nn.optimizer.get_lr()`：获取当前step的学习率，可以在自定义优化器时，在construct方法中使用。

- `mindspore.nn.optimizer.get_lr_parameter(params)`：获取指定参数组的参数学习率列表，如果是固定学习率，返回一个标量Parameter的列表；如果是计算图格式的动态学习率，返回一个Cell的列表；如果是列表格式的动态学习率，返回shape为(n,)的Parameter的列表（其中n是动态学习率列表的长度）。

#### 获取优化器的状态

`PyTorch.optimizer.param_groups`：获取优化器相关配置参数的状态，返回数据格式为字典的列表，key为参数名，value为参数值。以SGD为例，字典的key为key为'params'、 'lr'、'momentum'、'dampening'、'weight_decay'、 'nesterov'等。

`PyTorch.optimizer.state_dict`：获取optimizer的状态，返回一个key为“state”、“param_groups”，value为对应数值的字典。

MindSpore暂无对应功能。

## 自定义优化器

MindSpore和PyTorch都支持用户基于python基本语法及相关算子自定义优化器。在PyTorch中，通过重写`__init__`和`step`方法，用户可以根据需求自定义优化器，具体用法可以参考[这篇教程](http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html)。MindSpore也支持类似用法，以Momentum为例，使用基础的小算子构建：

```python
from mindspore import Parameter, ops, nn

class MomentumOpt(nn.Optimizer):
    def __init__(self, params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0, use_nesterov=False):
        super(MomentumOpt, self).__init__(learning_rate, params, weight_decay, loss_scale)
        self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
        self.moments = self.parameters.clone(prefix="moments", init="zeros")
        self.assign = ops.Assign()
    def construct(self, gradients):
        params = self.parameters
        moments = self.moments
        success = None
        for param, mom, grad in zip(params, moments, gradients):
            # 小算子表达
            update = self.momentum * param + mom + self.learning_rate * grad
            success = self.assign(param, update)
        return success
```

MindSpore的`ops`模块也提供了`ApplyMomentum`的高阶算子，使用方式可参考：

```python
from mindspore import Parameter, ops, nn

class MomentumOpt(nn.Optimizer):
    def __init__(self, params, learning_rate, momentum, weight_decay=0.0, loss_scale=1.0, use_nesterov=False):
        super(MomentumOpt, self).__init__(learning_rate, params, weight_decay, loss_scale)
        self.moments = self.parameters.clone(prefix="moments", init="zeros")
        self.opt = ops.ApplyMomentum(use_nesterov=use_nesterov)
    def construct(self, gradients):
        params = self.parameters
        moments = self.moments
        success = None
        for param, mom, grad in zip(params, moments, gradients):
          # 大算子表达
          success = self.opt(param, mom, self.learning_rate, grad, self.momentum)
        return success
```

## API映射

MindSpore和PyTorch的API对应关系和差异可以参考[API映射](https://www.mindspore.cn/docs/migration_guide/zh-CN/master/api_mapping/pytorch_api_mapping.html)，其余暂时没有对应关系的接口目前情况如下：

```python
# PyTorch
PyTorch.optim.ASGD
PyTorch.optim.LBFGS
```

```python
# mindspore
mindspore.nn.ProximalAadagrad
mindspore.nn.AdamOffload
mindspore.nn.FTRL
mindspore.nn.Lamb
mindspore.nn.thor
```
