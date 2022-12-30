# Comparing the function difference with torch.optim.lr_scheduler.ExponentialLR

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ExponentialDecayLR.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.optim.lr_scheduler.ExponentialLR

```python
torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma,
    last_epoch=-1,
    verbose=False
)
```

For more information, see [torch.optim.lr_scheduler.ExponentialLR](https://pytorch.org/docs/1.8.1/optim.html#torch.optim.lr_scheduler.ExponentialLR).

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

For more information, see [mindspore.nn.exponential_decay_lr](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.exponential_decay_lr.html#mindspore.nn.exponential_decay_lr).

## mindspore.nn.ExponentialDecayLR

```python
mindspore.nn.ExponentialDecayLR(
  learning_rate,
  decay_rate,
  decay_steps,
  is_stair=False
)
```

For more information, see [mindspore.nn.ExponentialDecayLR](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.ExponentialDecayLR.html#mindspore.nn.ExponentialDecayLR).

## Differences

PyTorch (torch.optim.lr_scheduler.ExponentialLR): The calculating method is $lr * gamma^{epoch}$ . When used, the optimizer is used as input and the learning rate is updated by calling the `step` method. When `verbose` is True, the relevant information is printed for each update.

MindSpore (mindspore.nn.exponential_decay_lr): The calculating method is $lr * decay\_rate^{p}$ . `exponential_decay_lr` pre-generates the learning rate list and passes the list into the optimizer.

| Categories | Subcategories  | PyTorch | MindSpore | Differences                 |
| ---- | ----- | ------- | --------- | -------------------- |
| Parameter  | Parameter 1 | optimizer   |        | Optimizer for PyTorch applications. MindSpore does not have this Parameter  |
|      | Parameter 2 | gamma |   decay_rate   | Parameter of decay learning rate, same function, different Parameter name |
|      | Parameter 3 | last_epoch |   | MindSpore does not have this Parameter  |
|      | Parameter 4 | verbose |        | PyTorch `verbose` prints information about each update when it is True. MindSpore does not have this Parameter.  |
|      | Parameter 5 |  | learning_rate | MindSpore sets the initial value of the learning rate. |
|      | Parameter 6 |  | total_step | Total number of steps in MindSpore |
|      | Parameter 7 |  | step_per_epoch | The number of steps per epoch in MindSpore |
|      | Parameter 8 |       |  decay_steps   | The number of decay steps performed by MindSpore |
|      | Parameter 9 |       |  is_stair   | When MindSpore `is_stair` is True, the learning rate decays once every `decay_steps`. |

MindSpore (mindspore.nn.ExponentialDecayLR): The calculating method is $lr * decay\_rate^{p}$ . `ExponentialDecayLR` is passed in the optimizer for training in the way of the computational graph.

| Categories | Subcategories  | PyTorch | MindSpore | Differences                |
| ---- | ----- | ------- | --------- | -------------------- |
| Parameter  | Parameter 1 | optimizer   |        | Optimizer for PyTorch applications. MindSpore does not have this Parameter  |
|      | Parameter 2 | gamma |   decay_rate   | Parameter of decay learning rate, same function, different Parameter name |
|      | Parameter 3 | last_epoch |   | MindSpore does not have this Parameter.  |
|      | Parameter 4 | verbose |        | PyTorch `verbose` prints information about each update when it is True. MindSpore does not have this Parameter. |
|      | Parameter 5 |  | learning_rate | MindSpore sets the initial value of the learning rate. |
|      | Parameter 6 |       |  decay_steps   | The number of decay steps performed by MindSpore |
|      | Parameter 7 |       |  is_stair   | When MindSpore `is_stair` is True, the learning rate decays once every `decay_steps`. |

## Code Example

```python
# In MindSpore：
import mindspore as ms
from mindspore import nn

# In MindSpore：exponential_decay_lr
learning_rate = 0.1
decay_rate = 0.9
total_step = 6
step_per_epoch = 2
decay_epoch = 1
output = nn.exponential_decay_lr(learning_rate, decay_rate, total_step, step_per_epoch, decay_epoch)
print(output)
# [0.1, 0.1, 0.09000000000000001, 0.09000000000000001, 0.08100000000000002, 0.08100000000000002]

# In MindSpore：ExponentialDecayLR
learning_rate = 0.1
decay_rate = 0.9
decay_steps = 4
global_step = ms.Tensor(2, ms.int32)
exponential_decay_lr = nn.ExponentialDecayLR(learning_rate, decay_rate, decay_steps)
result = exponential_decay_lr(global_step)
print(result)
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
