# Comparing the function differences between torch.optim.lr_scheduler.CosineAnnealingLR and torch.optim.lr_scheduler.cosine_decay_lr

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/CosineDecayLr.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see[torch.optim.lr_scheduler.CosineAnnealingLR](https://pytorch.org/docs/1.8.1/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR).

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

For more information, see[mindspore.nn.cosine_decay_lr](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.cosine_decay_lr.html#mindspore.nn.cosine_decay_lr).

## Differences

PyTorch (torch.optim.lr_scheduler.CosineAnnealingLR): `torch.optim.lr_scheduler.CosineAnnealingLR` is used to periodically adjust the learning rate, where the input parameter `T_max` represents 1/2 of the period. Assuming the initial learning rate is `lr`, in each period of `2*T_max`, the learning rate changes according to the specified calculation logic, for the formula detail, see the API docs; after the period ends, the learning rate returns to the initial value `lr` , and keep looping. When `verbose` is True, the relevant information is printed for each update.

MindSpore (mindspore.nn.cosine_decay_lr): the learning rate adjustment of `mindspore.nn.cosine_decay_lr` has no periodic changes, and the learning rate value changes according to the specified calculation logic. The formula calculation logic is the same as that of `torch.optim.lr_scheduler.CosineAnnealingLR`.

| Categories | Subcategories  | PyTorch | MindSpore | Differences                 |
| ---- | ----- | ------- | --------- | -------------------- |
| Parameter | Parameter 1 | optimizer   |        | Optimizer for PyTorch applications. MindSpore does not have this parameter |
|      | Parameter 2 | T_max   | total_step | The step to perform decay. The function is the same, and the parameter name is different |
|      | Parameter 3 | eta_min | min_lr     | Minimum learning rate, same function, different parameter names |
|      | Parameter 4 | last_epoch |  decay_epoch | The functions are the same, but the parameter names are different |
|      | Parameter 5 | verbose |        | PyTorch prints information about each update when `verbose` is True. MindSpore does not have this parameter |
|      | Parameter 6 |       |  max_lr   | Maximum learning rate. PyTorch is set to initial lr, and MindSpore is set to `max_lr` |
|      | Parameter 7 |       |  step_per_epoch   | MindSpore number of steps per epoch |

## Code Example

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
# [0.1, 0.1, 0.05500000000000001, 0.05500000000000001, 0.01, 0.01]


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
