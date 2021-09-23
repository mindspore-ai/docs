# 比较与torch.optim.lr_scheduler.StepLR和torch.optim.lr_scheduler.MultiStepLR的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/PiecewiseConstantLR.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.optim.lr_scheduler.StepLR

```python
torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size,
    gamma=0.1,
    last_epoch=-1,
    verbose=False
)
```

## torch.optim.lr_scheduler.MultiStepLR

```python
torch.optim.lr_scheduler.MultiStepLR(
     optimizer,
     milestones,
     gamma=0.1,
     last_epoch=-1,
     verbose=False)
)
```

## mindspore.nn.piecewise_constant_lr

```python
mindspore.nn.piecewise_constant_lr(
    milestone,
    learning_rates
)
```

## 使用方式

PyTorch: 分段设置学习率，`torch.optim.lr_scheduler.StepLR`通过传入step_size，每隔固定的step_size，学习率乘以gamma；`torch.optim.lr_scheduler.MultiStepLR`通过传入milestones的step数值列表，达到step数值，学习率乘以gamma。使用时，优化器作为输入，在训练过程中调用`step`方法进行数值的更新。

MindSpore：传入milestones的step数值列表和对应的学习率设置值列表，达到step数值，学习率取对应的值。最终返回一个学习率的列表，作为优化器的输入。

## 代码示例

```python
from mindspore import nn

# In MindSpore：
milestone = [2, 5, 10]
learning_rates = [0.1, 0.05, 0.01]
output = nn.piecewise_constant_lr(milestone, learning_rates)
print(output)
# Out：
# [0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]


# In torch:
import numpy as np
import torch
from torch import optim

model = torch.nn.Sequential(torch.nn.Linear(20, 1))
optimizer = optim.SGD(model.parameters(), 0.1)

# 使用step_lr
step_lr = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
# 使用multi_step_lr
multi_step_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.9)
```
