# Comparing the function differences between torch.optim.lr_scheduler.StepLR and torch.optim.lr_scheduler.MultiStepLR

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/PiecewiseConstantLR.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## torch.optim.lr_scheduler.StepLR

```python
torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size,
    gamma=0.1,
    last_epoch=-1
)
```

For more information, see [torch.optim.lr_scheduler.StepLR](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.lr_scheduler.StepLR).

## torch.optim.lr_scheduler.MultiStepLR

```python
torch.optim.lr_scheduler.MultiStepLR(
     optimizer,
     milestones,
     gamma=0.1,
     last_epoch=-1
)
```

For more information, see [torch.optim.lr_scheduler.MultiStepLR](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.lr_scheduler.MultiStepLR).

## mindspore.nn.piecewise_constant_lr

```python
mindspore.nn.piecewise_constant_lr(
    milestone,
    learning_rates
)
```

For more information, see [mindspore.nn.piecewise_constant_lr](https://mindspore.cn/docs/api/en/r1.6/api_python/nn/mindspore.nn.piecewise_constant_lr.html#mindspore.nn.piecewise_constant_lr).

## Differences

PyTorch: `torch.optim.lr_scheduler.StepLR`calculates the learning rate by step_size, value of lr will multiply gamma for every fixed step size; `torch.optim.lr_scheduler.MultiStepLR`ccalculates the learning rate by a step size list, value of lr will multiply gamma when the step size is reached in the list. During the training stage, the optimizer is passed to the lr scheduler, which calls the step method to update the current learning rate.

MindSpore: Step size list and the corresponding learning rate value list are passed to the function, and a list of learning rates will be returned as input to the optimizer.

## Code Example

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
# step_lr
step_lr = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
# multi_step_lr
multi_step_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.9)
```
