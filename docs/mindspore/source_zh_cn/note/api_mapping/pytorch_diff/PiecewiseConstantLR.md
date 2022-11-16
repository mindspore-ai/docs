# 比较与torch.optim.lr_scheduler.StepLR和torch.optim.lr_scheduler.MultiStepLR的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/PiecewiseConstantLR.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[torch.optim.lr_scheduler.StepLR](https://pytorch.org/docs/1.8.1/optim.html#torch.optim.lr_scheduler.StepLR)。

## torch.optim.lr_scheduler.MultiStepLR

```python
torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones,
    gamma=0.1,
    last_epoch=-1,
    verbose=False
)
```

更多内容详见[torch.optim.lr_scheduler.MultiStepLR](https://pytorch.org/docs/1.8.1/optim.html#torch.optim.lr_scheduler.MultiStepLR)。

## mindspore.nn.piecewise_constant_lr

```python
mindspore.nn.piecewise_constant_lr(
    milestone,
    learning_rates
)
```

更多内容详见[mindspore.nn.piecewise_constant_lr](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.piecewise_constant_lr.html#mindspore.nn.piecewise_constant_lr)。

## 差异对比

PyTorch（torch.optim.lr_scheduler.StepLR）：分段设置学习率，`torch.optim.lr_scheduler.StepLR`通过传入step_size，每隔固定的step_size，学习率乘以gamma， `verbose` 为True时，每一次更新打印相关信息。

MindSpore（mindspore.nn.piecewise_constant_lr）：传入milestones的step数值列表和对应的学习率设置值列表，达到step数值，学习率取对应的值。最终返回一个学习率的列表，作为优化器的输入。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | optimizer   |        | PyTorch应用的优化器，MindSpore无此参数 |
|      | 参数2 | step_size   | milestone | MindSpore分段更新学习率的step列表，PyTorch使用固定的step值 |
|      | 参数3 | gamma |      | PyTorch衰减学习率的参数，MindSpore无此参数 |
|      | 参数4 | last_epoch |   | MindSpore无此参数 |
|      | 参数5 | verbose |        | PyTorch的 `verbose` 为True时，每一次更新打印相关信息。MindSpore无此参数 |
|      | 参数6 |       |  learning_rates   | MindSpore设置学习率的列表 |

PyTorch（torch.optim.lr_scheduler.MultiStepLR）：`torch.optim.lr_scheduler.MultiStepLR`通过传入milestones的step数值列表，达到step数值，学习率乘以gamma。使用时，优化器作为输入，在训练过程中调用 `step` 方法进行数值的更新。 `verbose` 为True时，每一次更新打印相关信息。

MindSpore（mindspore.nn.piecewise_constant_lr）：传入milestones的step数值列表和对应的学习率设置值列表，达到step数值，学习率取对应的值。最终返回一个学习率的列表，作为优化器的输入。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | optimizer   |        | PyTorch应用的优化器，MindSpore无此参数 |
|      | 参数2 | milestones   | milestone | 分段更新学习率的step列表，功能一致，参数名不同 |
|      | 参数3 | gamma |      | PyTorch衰减学习率的参数，MindSpore无此参数 |
|      | 参数4 | last_epoch |   | MindSpore无此参数 |
|      | 参数5 | verbose |        | PyTorch的 `verbose` 为True时，每一次更新打印相关信息。MindSpore无此参数 |
|      | 参数6 |       |  learning_rates   | MindSpore设置学习率的列表 |

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
