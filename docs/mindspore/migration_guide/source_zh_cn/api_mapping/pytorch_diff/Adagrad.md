# 比较与torch.optim.Adagrad的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Adagrad.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.optim.Adagrad

```python
class torch.optim.Adagrad(
    params,
    lr=0.01,
    lr_decay=0,
    weight_decay=0,
    initial_accumulator_value=0,
    eps=1e-10
)
```

## mindspore.nn.Adagrad

```python
class mindspore.nn.Adagrad(
    params,
    accum=0.1,
    learning_rate=0.001,
    update_slots=True,
    loss_scale=1.0,
    weight_decay=0.0
)(grads)
```

## 使用方式

PyTorch: 需要将期望更新的参数放入1个迭代类型参数`params`后传入，且设置了`step`方法执行单步优化返回损失值。

MindSpore：支持所有的参数使用相同的学习率以及不同的参数组使用不用的值的方式。

## 代码示例

```python
# The following implements Adagrad with MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import Model
from mindspore import dtype as mstype

net = Net()
#1) All parameters use the same learning rate and weight decay
optim = nn.Adagrad(params=net.trainable_params())

#2) Use parameter groups and set different values
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}]
optim = nn.Adagrad(group_params, learning_rate=0.1, weight_decay=0.0)
# The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
# centralization of True.
# The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
# centralization of False.
# The final parameters order in which the optimizer will be followed is the value of 'order_params'.

loss = nn.SoftmaxCrossEntropyWithLogits()
model = Model(net, loss_fn=loss, optimizer=optim)

# The following implements Adagrad with torch.
input_x = torch.tensor(np.random.rand(1, 20).astype(np.float32))
input_y = torch.tensor([1.])
net = torch.nn.Sequential(torch.nn.Linear(input_x.shape[-1], 1))
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adagrad(net.parameters())
l = loss(net(input_x).view(-1), input_y) / 2
optimizer.zero_grad()
l.backward()
optimizer.step()
print(loss(net(input_x).view(-1), input_y).item() / 2)
# Out:
# 0.1830
```
