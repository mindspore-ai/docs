# Comparing the function differences between torch.optim.lr_scheduler.StepLR and torch.optim.lr_scheduler.MultiStepLR

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/pytorch_diff/PiecewiseConstantLR.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

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

For more information, see [torch.optim.lr_scheduler.StepLR](https://pytorch.org/docs/1.8.1/optim.html#torch.optim.lr_scheduler.StepLR).

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

For more information, see [mindspore.nn.piecewise_constant_lr](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/nn/mindspore.nn.piecewise_constant_lr.html#mindspore.nn.piecewise_constant_lr).

## Differences

PyTorch (torch.optim.lr_scheduler.StepLR): Setting the learning rate in segments. `torch.optim.lr_scheduler.StepLR` multiplies the learning rate by gamma every fixed step_size, by passing in step_size. When `verbose` is True, the relevant information is printed for each update.

MindSpore (mindspore.nn.piecewise_constant_lr): Pass in the list of step values of milestones and the corresponding list of learning rate setting values, reach the step values and the learning rate takes the corresponding values. A list of learning rates is eventually returned as input to the optimizer.

| Categories | Subcategories  | PyTorch | MindSpore | Differences                 |
| ---- | ----- | ------- | --------- | -------------------- |
| Parameter  | Parameter 1 | optimizer   |        | Optimizer for PyTorch applications. MindSpore does not have this Parameter  |
|      | Parameter 2 | step_size |   milestone   | MindSpore segmentation updates the step list of learning rates, and PyTorch uses a fixed step value. |
|      | Parameter 3 | gamma |   | Parameters for PyTorch decay learning rate. MindSpore does not have this Parameter.  |
|      | Parameter 4 | last_epoch |        | MindSpore does not have this Parameter. |
|      | Parameter 5 |  | verbose | PyTorch `verbose` prints information about each update when it is True. |
|      | Parameter 6 |       |  learning_rates   | List of MindSpore settings for learning rate |

PyTorch (torch.optim.lr_scheduler.MultiStepLR): `torch.optim.lr_scheduler.MultiStepLR` reaches the step value by passing in a list of step values for milestones, and the learning rate is multiplied by gamma. When used, the optimizer is used as input and the `step` method is called during the training process to update the values. `verbose` prints information about each update when it is True.

MindSpore (mindspore.nn.piecewise_constant_lr): Pass in the list of step values for milestones and the corresponding list of learning rate setting values, reach the step values and the learning rate takes the corresponding values. A list of learning rates is eventually returned as input to the optimizer.

| Categories | Subcategories  | PyTorch | MindSpore | Differences                 |
| ---- | ----- | ------- | --------- | -------------------- |
| Parameter  | Parameter 1 | optimizer   |        | Optimizer for PyTorch applications. MindSpore does not have this Parameter  |
|      | Parameter 2 | milestones |   milestone   | Segmentation updates the step list of learning rates, with the same functions and different parameter names |
|      | Parameter 3 | gamma |   | Parameters for PyTorch decay learning rate. MindSpore does not have this Parameter.  |
|      | Parameter 4 | last_epoch |        | MindSpore does not have this Parameter. |
|      | Parameter 5 |  | verbose | PyTorch `verbose` prints information about each update when it is True. MindSpore does not have this Parameter.|
|      | Parameter 6 |       |  learning_rates   | List of MindSpore settings for learning rate |

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
