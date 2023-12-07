# 学习率与优化器

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/migration_guide/model_development/learning_rate_and_optimizer.md)

在阅读本章节之前，请先阅读MindSpore官网教程[优化器](https://mindspore.cn/tutorials/zh-CN/r2.3/advanced/modules/optimizer.html)。

这里就MindSpore的优化器的一些特殊使用方式和学习率衰减策略的原理做一个介绍。

## 优化器对比

### 优化器支持差异

PyTorch和MindSpore同时支持的优化器异同比较详见[API映射表](https://mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html#torch-optim)。MindSpore暂不支持的优化器：LBFGS，NAdam，RAdam。

### 优化器的执行和使用差异

PyTorch单步执行优化器时，一般需要手动执行 `zero_grad()` 方法将历史梯度设置为0(或None)，然后使用 `loss.backward()` 计算当前训练step的梯度，最后调用优化器的 `step()` 方法实现网络权重的更新；

MindSpore中优化器的使用，只需要直接对梯度进行计算，然后使用 `optimizer(grads)` 执行网络权重的更新。

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
import mindspore
from mindspore import nn

optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01)
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss
```

</pre>
</td>
</tr>
</table>
</div>

### 超参差异

#### 超参名称

网络权重和学习率入参名称异同：

| 参数   | PyTorch | MindSpore | 差异    |
|------|---------| --------- |-------|
| 网络权重 | params  | params      | 参数名相同 |
| 学习率  | lr      | learning_rate      | 参数名不同 |

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
from torch import optim

optimizer = optim.SGD(model.parameters(), lr=0.01)
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
from mindspore import nn

optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01)
```

</pre>
</td>
</tr>
</table>
</div>

#### 超参配置方式

- 参数不分组：

    `params` 入参支持类型不同： PyTorch入参类型为 `iterable(Tensor)` 和 `iterable(dict)`，支持迭代器类型；
MindSpore入参类型为 `list(Parameter)`，`list(dict)`，不支持迭代器。

    其他超参配置及支持差异详见[API映射表](https://mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html#torch-optim)。

- 参数分组：

    PyTorch支持所有参数分组；MindSpore仅支持特定key分组："params"，"lr"，"weight_decay"，"grad_centralization"，"order_params"。

    <div class="wy-table-responsive">
    <table class="colwidths-auto docutils align-default">
    <tr>
    <td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
    </tr>
    <tr>
    <td style="vertical-align:top"><pre>

    ```python
    optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
    ```

    </pre>
    </td>
    <td style="vertical-align:top"><pre>

    ```python
    conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
    no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
    group_params = [{'params': conv_params, 'weight_decay': 0.01, 'lr': 0.02},
            {'params': no_conv_params}]

    optim = nn.Momentum(group_params, learning_rate=0.1, momentum=0.9)
    ```

    </pre>
    </td>
    </tr>
    </table>
    </div>

#### 运行时超参修改

PyTorch支持在训练过程中修改任意的优化器参数，并提供了 `LRScheduler` 用于动态修改学习率；

MindSpore当前不支持训练过程中修改优化器参数，但提供了修改学习率和权重衰减的方式，使用方式详见[学习率](#学习率策略对比)和[权重衰减](#权重衰减)章节。

### 权重衰减

PyTorch中修改 `weight_decay` 示例如下；

MindSpore中实现动态weight decay：用户可以继承 `Cell` 自定义动态weight decay的类，传入优化器中。

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
from torch.nn import optim

optimizer = optim.SGD(param_groups, lr=0.01, weight_decay=0.1)
decay_factor = 0.1
def train_step(data, label):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, label)
    loss.backward()
    optimizer.step()
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] *= decay_factor
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
class ExponentialWeightDecay(Cell):

    def __init__(self, weight_decay, decay_rate, decay_steps):
        super(ExponentialWeightDecay, self).__init__()
        self.weight_decay = weight_decay
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def construct(self, global_step):
        p = global_step / self.decay_steps
        return self.weight_decay * ops.pow(self.decay_rate, p)

weight_decay = ExponentialWeightDecay(weight_decay=0.1, decay_rate=0.1, decay_steps=10000)
optimizer = nn.SGD(net.trainable_params(), weight_decay=weight_decay)
```

</pre>
</td>
</tr>
</table>
</div>

### 优化器状态的保存与加载

PyTorch的优化器模块提供了 `state_dict()` 用于优化器状态的查看及保存，`load_state_dict` 用于优化器状态的加载。

MindSpore的优化器模块继承自 `Cell`，优化器的保存与加载和网络的保存与加载方式相同，通常情况下配合 `save_checkpoint` 与`load_checkpoint` 使用。

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
# 优化器保存：
# 使用torch.save()把获取到的state_dict保存到pkl文件中
optimizer = optim.SGD(param_groups, lr=0.01)
torch.save(optimizer.state_dict(), save_path)
```

```python
# 优化器加载：
# 使用torch.load()加载保存的state_dict，
# 然后使用load_state_dict将获取到的state_dict加载到优化器中
optimizer = optim.SGD(param_groups, lr=0.01)
state_dict = torch.load(save_path)
optimizer.load_state_dict(state_dict)
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
# 优化器保存：
# 使用mindspore.save_checkpoint()将优化器实例保存到ckpt文件中
optimizer = nn.SGD(param_groups, lr=0.01)
state_dict = mindspore.save_checkpoint(opt, save_path)
```

```python
# 优化器加载：
# 使用mindspore.load_checkpoint()加载保存的ckpt文件，
# 然后使用load_param_into_net将获取到的param_dict加载到优化器中
optimizer = nn.SGD(param_groups, lr=0.01)
param_dict = mindspore.load_checkpoint(save_path)
mindspore.load_param_into_net(opt, param_dict)
```

</pre>
</td>
</tr>
</table>
</div>

## 学习率策略对比

### 动态学习率差异

PyTorch中定义了 `LRScheduler` 类用于对学习率进行管理。使用动态学习率时，将 `optimizer` 实例传入 `LRScheduler` 子类中，通过循环调用 `scheduler.step()` 执行学习率修改，并将修改同步至优化器中。

MindSpore中的动态学习率有 `Cell` 和 `list` 两种实现方式，两种类型的动态学习率使用方式一致，都是在实例化完成之后传入优化器，前者在内部的 `construct` 中进行每一步学习率的计算，后者直接按照计算逻辑预生成学习率列表，训练过程中内部实现学习率的更新。具体请参考[动态学习率](https://mindspore.cn/docs/zh-CN/r2.3/api_python/mindspore.nn.html#%E5%8A%A8%E6%80%81%E5%AD%A6%E4%B9%A0%E7%8E%87)。

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
polynomial_decay_lr = nn.PolynomialDecayLR(learning_rate=0.1, end_learning_rate=0.01, decay_steps=4, power=0.5)
optim = nn.Momentum(params, learning_rate=polynomial_decay_lr, momentum=0.9, weight_decay=0.0)

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss
```

</pre>
</td>
</tr>
</table>
</div>

### 自定义学习率差异

PyTorch的动态学习率模块 `LRScheduler` 提供了`LambdaLR` 接口供用户自定义学习率调整规则，用户通过传入lambda表达式或自定义函数实现学习率指定。

MindSpore未提供类似的lambda接口，自定义学习率调整策略可以通过自定义函数或自定义 `LearningRateSchedule` 来实现。

<div class="wy-table-responsive">
<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
lbd = lambda epoch: epoch // 5
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lbd)

for epoch in range(20):
    train(...)
    validate(...)
    scheduler.step()
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
# 方式一：定义python函数指定计算逻辑，返回学习率列表
def dynamic_lr(lr, total_step, step_per_epoch):
    lrs = []
    for i in range(total_step):
        current_epoch = i // step_per_epoch
        factor = current_epoch // 5
        lrs.append(lr * factor)
    return lrs

decay_lr = dynamic_lr(lr=0.01, total_step=200, step_per_epoch=10)
optim = nn.SGD(params, learning_rate=decay_lr)
```

```python
# 方式二：继承LearningRateSchedule，在construct方法中定义变化策略
class DynamicDecayLR(LearningRateSchedule):
    def __init__(self, lr, step_per_epoch):
        super(DynamicDecayLR, self).__init__()
        self.lr = lr
        self.step_per_epoch = step_per_epoch
        self.cast = P.Cast()

    def construct(self, global_step):
        current_epoch = self.cast(global_step, mstype.float32) // step_per_epoch
        return self.learning_rate * (current_epoch // 5)

decay_lr = DynamicDecayLR(lr=0.01, step_per_epoch=10)
optim = nn.SGD(params, learning_rate=decay_lr)
```

</pre>
</td>
</tr>
</table>
</div>

### 学习率获取

PyTorch：

- 固定学习率情况下，通常通过 `optimizer.state_dict()` 进行学习率的查看和打印，例如参数分组时，对于第n个参数组，使用 `optimizer.state_dict()['param_groups'][n]['lr']`，参数不分组时，使用 `optimizer.state_dict()['param_groups'][0]['lr']`；

- 动态学习率情况下，可以使用 `LRScheduler` 的 `get_lr` 方法获取当前学习率，或使用 `print_lr` 方法打印学习率。

MindSpore：

- 目前未提供直接查看学习率的接口，后续版本中会针对此问题进行修复。

### 学习率更新

PyTorch：

PyTorch提供了`torch.optim.lr_scheduler`包用于动态修改lr，使用的时候需要显式地调用`optimizer.step()`和`scheduler.step()`来更新lr，详情请参考[如何调整学习率](https://pytorch.org/docs/1.12/optim.html#how-to-adjust-learning-rate)。

MindSpore：

MindSpore的学习率是包到优化器里面的，每调用一次优化器，学习率更新的step会自动更新一次。

## 参数分组

MindSpore的优化器支持一些特别的操作，比如对网络里所有的可训练的参数可以设置不同的学习率（lr）、权重衰减（weight_decay）和梯度中心化（grad_centralization）策略，如：

```python
from mindspore import nn

# 定义模型
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.SequentialCell([
            nn.Conv2d(3, 12, kernel_size=3, pad_mode='pad', padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.layer2 = nn.SequentialCell([
            nn.Conv2d(12, 4, kernel_size=3, pad_mode='pad', padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.pool = nn.AdaptiveMaxPool2d((5, 5))
        self.fc = nn.Dense(100, 10)

    def construct(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view((-1, 100))
        out = nn.Dense(x)
        return out

def params_not_in(param, param_list):
    # 利用Parameter的id来判断一个param是否不在param_list中
    param_id = id(param)
    for p in param_list:
        if id(p) == param_id:
            return False
    return True

net = Network()
trainable_param = net.trainable_params()
conv_weight, bn_weight, dense_weight = [], [], []
for _, cell in net.cells_and_names():
    # 判断是什么API，将对应参数加到不同列表里
    if isinstance(cell, nn.Conv2d):
        conv_weight.append(cell.weight)
    elif isinstance(cell, nn.BatchNorm2d):
        bn_weight.append(cell.gamma)
        bn_weight.append(cell.beta)
    elif isinstance(cell, nn.Dense):
        dense_weight.append(cell.weight)

other_param = []
# 所有分组里的参数不能重复，并且其交集是需要做参数更新的所有参数
for param in trainable_param:
    if params_not_in(param, conv_weight) and params_not_in(param, bn_weight) and params_not_in(param, dense_weight):
        other_param.append(param)

group_param = [{'order_params': trainable_param}]
# 每一个分组的参数列表不能是空的

if conv_weight:
    conv_weight_lr = nn.cosine_decay_lr(0., 1e-3, total_step=1000, step_per_epoch=100, decay_epoch=10)
    group_param.append({'params': conv_weight, 'weight_decay': 1e-4, 'lr': conv_weight_lr})
if bn_weight:
    group_param.append({'params': bn_weight, 'weight_decay': 0., 'lr': 1e-4})
if dense_weight:
    group_param.append({'params': dense_weight, 'weight_decay': 1e-5, 'lr': 1e-3})
if other_param:
    group_param.append({'params': other_param})

opt = nn.Momentum(group_param, learning_rate=1e-3, weight_decay=0.0, momentum=0.9)
```

需要注意以下几点：

1. 每一个分组的参数列表不能是空的；
2. 如果没有设置`weight_decay`和`lr`则使用优化器里设置的值，设置了的话使用分组参数字典里的值；
3. 每个分组里的`lr`都可以是静态或动态的，但不能再分组；
4. 每个分组里的`weight_decay`都需要是符合规范的浮点数；
5. 所有分组里的参数不能重复，并且其交集是需要做参数更新的所有参数。

## MindSpore的学习率衰减策略

在训练过程中，MindSpore的学习率是以参数的形式存在于网络里的，在执行优化器更新网络可训练参数前，MindSpore会调用[get_lr](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/nn/mindspore.nn.Optimizer.html#mindspore.nn.Optimizer.get_lr)
方法获取到当前step需要的学习率的值。

MindSpore的学习率支持静态、动态、分组三种，其中静态学习率在网络里是一个float32类型的Tensor。

动态学习率有两种，一种在网络里是一个长度为训练总的step数，float32类型的Tensor，如[Dynamic LR函数](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/mindspore.nn.html#dynamic-lr%E5%87%BD%E6%95%B0)。在优化器里有一个`global_step`的参数，每经过一次优化器更新参数会+1，MindSpore内部会根据`global_step`和`learning_rate`这两个参数来获取当前step的学习率的值；
另一种是通过构图来生成学习率的值的，如[LearningRateSchedule类](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/mindspore.nn.html#learningrateschedule%E7%B1%BB)。

分组学习率如上一小节参数分组中介绍的。

因为MindSpore的学习率是参数，我们也可以通过给`learning_rate`参数赋值的方式修改训练过程中学习率的值，如[LearningRateScheduler Callback](https://www.mindspore.cn/docs/zh-CN/r2.3/_modules/mindspore/train/callback/_lr_scheduler_callback.html#LearningRateScheduler)，这种方法只支持优化器中传入静态的学习率。关键代码如下：

```python
import mindspore as ms
from mindspore import ops, nn

net = nn.Dense(1, 2)
optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
print(optimizer.learning_rate.data.asnumpy())
new_lr = 0.01
# 改写learning_rate参数的值
ops.assign(optimizer.learning_rate, ms.Tensor(new_lr, ms.float32))
print(optimizer.learning_rate.data.asnumpy())
```

运行结果：

```text
0.1
0.01
```
