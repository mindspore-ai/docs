# Comparing the function difference with torch.optim.lr_scheduler.ExponentialLR

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ExponentialDecayLR.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## torch.optim.lr_scheduler.ExponentialLR

```python
torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma,
    last_epoch=-1
)
```

For more information, see [torch.optim.lr_scheduler.ExponentialLR](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.lr_scheduler.ExponentialLR).

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

For more information, see [mindspore.nn.exponential_decay_lr](https://mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.exponential_decay_lr.html#mindspore.nn.exponential_decay_lr).

## mindspore.nn.ExponentialDecayLR

```python
mindspore.nn.ExponentialDecayLR(
  learning_rate,
  decay_rate,
  decay_steps,
  is_stair=False
)
```

For more information, see [mindspore.nn.ExponentialDecayLR](https://www.mindspore.cn/docs/en/r1.7/api_python/nn/mindspore.nn.ExponentialDecayLR.html#mindspore.nn.ExponentialDecayLR).

## Differences

PyTorch: The function of calculating the learning rate for each step is lr*gamma^{epoch}. In the training stage, the optimizer should be passed into the lr scheduler, then the step method will be implemented.

MindSpore: The function of calculating learning rate for each step is lr*decay_rate^{p}, which is implemented in two ways in MindSpore: `exponential_decay_lr` pregenerates a list of learning rates and passes the list to the optimizer; secondly, `ExponentialDecayLR` instance is passed into the optimizer, during the training process, the optimizer calls the instance taking the current step as the input to get the current learning rate.

## Code Example

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
