# 比较与torch.optim.lr_scheduler.CosineAnnealingLR的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/CosineDecayLr.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.optim.lr_scheduler.CosineAnnealingLR

```python
torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max,
    eta_min=0,
    last_epoch=-1,
    verbose=False
)
```

更多内容详见[torch.optim.lr_scheduler.CosineAnnealingLR](https://pytorch.org/docs/1.8.1/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR)。

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

更多内容详见[mindspore.nn.cosine_decay_lr](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.cosine_decay_lr.html#mindspore.nn.cosine_decay_lr)。

## 差异对比

PyTorch（torch.optim.lr_scheduler.CosineAnnealingLR）：`torch.optim.lr_scheduler.CosineAnnealingLR` 用来周期性地调整学习率，其中入参 `T_max` 表示周期的1/2。假设初始的学习率为 `lr`，在每个 `2*T_max` 的一个周期内，学习率根据指定计算逻辑进行变化，公式详见API注释；周期结束后，学习率恢复初始值 `lr`，并不断循环。 `verbose` 为True时，每一次更新打印相关信息。

MindSpore（mindspore.nn.cosine_decay_lr）：`mindspore.nn.cosine_decay_lr` 的学习率调整无周期性变化，学习率值按照指定的计算逻辑变化，公式计算逻辑与 `torch.optim.lr_scheduler.CosineAnnealingLR` 的相同。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | optimizer   |        | PyTorch应用的优化器，MindSpore无此参数 |
|      | 参数2 | T_max   | total_step | 进行衰减的step，功能一致，参数名不同 |
|      | 参数3 | eta_min | min_lr     | 学习率最小值，功能一致，参数名不同 |
|      | 参数4 | last_epoch | decay_epoch | 功能一致，参数名不同 |
|      | 参数5 | verbose |        | PyTorch的 `verbose` 为True时，每一次更新打印相关信息。MindSpore无此参数 |
|      | 参数6 |       |  max_lr   | 最大学习率，PyTorch设置为初始lr，MindSpore设置为 `max_lr` |
|      | 参数7 |    - |  step_per_epoch     | MindSpore每个epoch的step数 |

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
