# 比较与torch.optim.lr_scheduler.ExponentialLR的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ExponentialDecayLR.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.optim.lr_scheduler.ExponentialLR

```python
torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma,
    last_epoch=-1
)
```

更多内容详见[torch.optim.lr_scheduler.ExponentialLR](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.lr_scheduler.ExponentialLR)。

## mindspore.nn.exponential_decay_lr

```python
mindspore.nn.exponential_decay_lr(
      learning_rate,
      decay_rate,
      total_step,
      step_per_epoch,
      decay_epoch,
      is_stair=False
)
```

更多内容详见[mindspore.nn.exponential_decay_lr](https://mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.exponential_decay_lr.html#mindspore.nn.exponential_decay_lr)。

## mindspore.nn.ExponentialDecayLR

```python
mindspore.nn.ExponentialDecayLR(
  learning_rate,
  decay_rate,
  decay_steps,
  is_stair=False
)
```

更多内容详见[mindspore.nn.ExponentialDecayLR](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.ExponentialDecayLR.html#mindspore.nn.ExponentialDecayLR)。

## 使用方式

PyTorch：计算方式为lr*gamma^{epoch}。使用时，优化器作为输入，通过调用`step`方法进行学习率的更新。

MindSpore：计算方式为lr*decay_rate^{p}，这种动态学习率的调整方式在mindspore里有两种实现方式：`exponential_decay_lr`预生成学习率列表，将列表传入优化器；`ExponentialDecayLR`则是通过计算图的方式传入优化器中参与训练。

## 代码示例

```python
# In MindSpore：
from mindspore import nn, Tensor
from mindspore import dtype as mstype

# In MindSpore：exponential_decay_lr
learning_rate = 0.1
decay_rate = 0.9
total_step = 6
step_per_epoch = 2
decay_epoch = 1
output = nn.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch)
print(output)
# out
# [0.1, 0.1, 0.09000000000000001, 0.09000000000000001, 0.08100000000000002, 0.08100000000000002]

# In MindSpore：ExponentialDecayLR
learning_rate = 0.1
decay_rate = 0.9
decay_steps = 4
global_step = Tensor(2, mstype.int32)
exponential_decay_lr = nn.ExponentialDecayLR(learning_rate, decay_rate, decay_steps)
result = exponential_decay_lr(global_step)
print(result)
#  out
# 0.09486833

# In torch:
import torch
import numpy as np
from torch import optim

model = torch.nn.Sequential(torch.nn.Linear(20, 1))
optimizer = optim.SGD(model.parameters(), 0.1)
exponential_decay_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
myloss = torch.nn.MSELoss()
dataset = [(torch.tensor(np.random.rand(1, 20).astype(np.float32)), torch.tensor([1.]))]

for epoch in range(5):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = myloss(output.view(-1), target)
        loss.backward()
        optimizer.step()
    exponential_decay_lr.step()
    print(exponential_decay_lr.get_last_lr())
#  out
# [0.09000000000000001]
# [0.08100000000000002]
# [0.07290000000000002]
# [0.06561000000000002]
# [0.05904900000000002]
```
