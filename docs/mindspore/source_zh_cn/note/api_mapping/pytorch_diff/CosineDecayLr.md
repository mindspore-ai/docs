# 比较与torch.optim.lr_scheduler.CosineAnnealingLR的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/CosineDecayLr.md)

## torch.optim.lr_scheduler.CosineAnnealingLR

```python
torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max,
    eta_min=0,
    last_epoch=-1
)
```

更多内容详见[torch.optim.lr_scheduler.CosineAnnealingLR](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR)。

## mindspore.nn.cosine_decay_lr

```python
mindspore.nn.cosine_decay_lr(
    min_lr,
    max_lr,
    total_step,
    step_per_epoch,
    decay_epoch
)
```

更多内容详见[mindspore.nn.cosine_decay_lr](https://www.mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.cosine_decay_lr.html#mindspore.nn.cosine_decay_lr)。

## 使用方式

`torch.optim.lr_scheduler.CosineAnnealingLR` 用来周期性得调整学习率，其中入参 `T_max` 表示周期的1/2。假设初始的学习率为 `lr`，在每个 `2*T_max` 的一个周期内，学习率根据指定计算逻辑进行变化，公式详见API注释；周期结束后，学习率恢复初始值 `lr`，并不断循环。

`mindspore.nn.cosine_decay_lr` 的学习率调整无周期性变化，学习率值按照指定的计算逻辑变化，公式计算逻辑与 `torch.optim.lr_scheduler.CosineAnnealingLR` 的相同。

## 代码示例

```python
# In MindSpore：
import mindspore.nn as nn

min_lr = 0.01
max_lr = 0.1
total_step = 6
step_per_epoch = 2
decay_epoch = 2
output = nn.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch)
print(output)
# out: [0.1, 0.1, 0.05500000000000001, 0.05500000000000001, 0.01, 0.01]


# In PyTorch:
import torch
import numpy as np
from torch import optim

model = torch.nn.Sequential(torch.nn.Linear(20, 1))
optimizer = optim.SGD(model.parameters(), 0.1)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=0.002)


myloss = torch.nn.MSELoss()
dataset = [(torch.tensor(np.random.rand(1, 20).astype(np.float32)), torch.tensor([1.]))]

for epoch in range(6):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = myloss(output.view(-1), target)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(scheduler.get_last_lr())
# out:
# [0.002]
# [0.1]
# [0.002]
# [0.1]
# [0.002]
# [0.1]
```
