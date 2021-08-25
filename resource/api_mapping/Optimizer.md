# 比较与torch.optim.optimizer的入参设置的差异

```python
class torch.optim.Optimizer(
    params,
    defaults
)
```

## mindspore.nn.Optimizer()

```python
class mindspore.nn.Optimizer(
    learning_rate,
    parameters,
    weight_decay=0.0,
    loss_scale=1.0
)
```

## 使用方式

## parameters设置

- **使用默认接口**

MindSpore：optimizer的入参`params`直接使用`trainable_params`方法配置。

```python
from mindspore import nn

net = Net()
optim_sgd = nn.SGD(params=net.trainable_params())
```

PyTorch：optimizer的入参`params`直接使用`parameters`方法配置。

```python
from torch import optim

net = Net()
optim_sgd = optim.SGD(params=net.parameters(), lr=0.01)
```

- **用户自定义配置传入的参数**

MindSpore：首先使用`get_parameters`方法获取网络中所有的参数，再根据需要，例如参数名称等，进行筛选，并传入优化器中。

```python
from mindspore import nn

net = Net()
all_params = net.get_parameters()
no_conv_params = list(filter(lambda x: "conv" not in x.name, all_params))
optim_sgd = nn.SGD(no_conv_params)
```

PyTorch：首先使用`named_parameters`方法获取网络中所有的参数，再根据需要，例如参数名称等，进行筛选，并传入优化器中。

```python
from torch import optim

net = Net()
all_params = net.named_parameters()
no_conv_params = []
for pname, p in all_params:
  if "conv" not in pname:
    no_conv_params.append(p)
optim_sgd = optim.SGD(no_conv_params, lr=0.01)
```

## learning_rate设置

固定学习率：用法相同

动态学习率

mindspore：

```python
from mindspore import nn
from mindspore import Tensor
from mindspore import dtype as mstype

# dynamic_lr
milestone = [2, 5, 10]
learning_rates = [0.1, 0.05, 0.01]
lr_dynamic = nn.dynamic_lr.piecewise_constant_lr(milestone, learning_rates)
print(lr_dynamic)

# learning_rate_schedule
lr_schedule = nn.learning_rate_schedule.PolynomialDecayLR(learning_rate=0.1,
                                   end_learning_rate=0.01,
                                   decay_steps=4,
                                   power=0.5 )

global_step = Tensor(2, mstype.int32)
result = lr_schedule(global_step)
print(result)

# 用作优化器入参
optimizer1 = nn.Momentum(net.trainable_params(), learning_rate=lr_dynamic, momentum=0.9, weight_decay=0.9)
optimizer2 = nn.Momentum(net.trainable_params(), learning_rate=lr_schedule, momentum=0.9, weight_decay=0.9)
```

```python
[0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
0.0736396
```

pytorch：

```python
from torch import optim

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

## weight_decay设置

用法一致。

## loss_scale设置

- mindspore：作为优化器的入参，配合FixedLossScaleManager使用。
- pytorch：不单独为优化器设置混合精度。

## 参数分组

PyTorch和MindSpore都支持对分组的参数设置不同的值，基本用法相似。但mindspore只支持'params'，'weight_decay'，'lr'，'grad_centralizaiton'分组；pytorch支持优化器的所有参数分组。

>当前Mindspore和pytorch都存在个别优化器不支持参数分组，详情参考各优化器的说明。

MindSpore：

```python
from mindspore import nn

net = Net()

conv_params = list(filter(=lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: "conv" not in x.name, net.trainable_params()))

fix_lr = 0.01
polynomial_decay_lr = nn.learning_rate_schedule.PolynomialDecayLR(learning_rate=0.1,
                                   end_learning_rate=0.01,
                                   decay_steps=4,
                                   power=0.5 )

group_params = [{'params': conv_params, 'weight_decay': 0.01, 'lr': fix_lr},
                {'params': no_conv_params, 'lr': polynomial_decay_lr},
                {'order_params': net.trainable_params()}]

optim_sgd = nn.SGD(group_params, learning_rate=0.1)
```

PyTorch:

```python
from torch import optim

net = Net()

all_params = net.parameters()
conv_params = []
no_conv_params = []

for pname, p in net.named_parameters():
  if 'conv' in pname:
    conv_params += [p]
  else:
    no_conv_params += [p]

group_params = [{'params': conv_params, 'weight_decay': 0.01, 'lr': fix_lr},
                    {'params': no_conv_params, 'nesterov'=True}]

optim_sgd = optim.SGD(group_params, lr=0.01)
```
