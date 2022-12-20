# 比较与torch.optim.lr_scheduler.ExponentialLR的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ExponentialDecayLR.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.optim.lr_scheduler.ExponentialLR

```python
torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma,
    last_epoch=-1,
    verbose=False
)
```

更多内容详见[torch.optim.lr_scheduler.ExponentialLR](https://pytorch.org/docs/1.8.1/optim.html#torch.optim.lr_scheduler.ExponentialLR)。

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

更多内容详见[mindspore.nn.exponential_decay_lr](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.exponential_decay_lr.html#mindspore.nn.exponential_decay_lr)。

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

## 差异对比

PyTorch（torch.optim.lr_scheduler.ExponentialLR）：计算方式为 :math:`lr * gamma^{epoch}` 。使用时，优化器作为输入，通过调用 `step` 方法进行学习率的更新。 `verbose` 为True时，每一次更新打印相关信息。

MindSpore（mindspore.nn.exponential_decay_lr）：计算方式为 :math:`lr * decay\_rate^{p}` ， `exponential_decay_lr` 预生成学习率列表，将列表传入优化器。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | optimizer   |     -   | PyTorch应用的优化器，MindSpore无此参数 |
|      | 参数2 | gamma |   decay_rate   | 衰减学习率的参数，功能一致，参数名不同 |
|      | 参数3 | last_epoch |  - | MindSpore无此参数 |
|      | 参数4 | verbose |   -     | PyTorch `verbose` 为True时，每一次更新打印相关信息。MindSpore无此参数 |
|      | 参数5 |  | learning_rate | MindSpore设置学习率的初始值 |
|      | 参数6 |  | total_step | MindSpore的step总数 |
|      | 参数7 |  | step_per_epoch | MindSpore每个epoch的step数 |
|      | 参数8 |    -   |  decay_steps   | MindSpore进行衰减的step数 |
|      | 参数9 |  -     |  is_stair   | MindSpore `is_stair` 为True时，学习率每 `decay_steps` 衰减一次 |

MindSpore（mindspore.nn.ExponentialDecayLR）：计算方式为 :math:`lr * decay\_rate^{p}` ， `ExponentialDecayLR` 是通过计算图的方式传入优化器中参与训练。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | optimizer   |   -     | PyTorch应用的优化器，MindSpore无此参数 |
|      | 参数2 | gamma |   decay_rate   | 衰减学习率的参数，功能一致，参数名不同 |
|      | 参数3 | last_epoch |  - | MindSpore无此参数 |
|      | 参数4 | verbose |   -     | PyTorch的 `verbose` 为True时，每一次更新打印相关信息。MindSpore无此参数 |
|      | 参数5 |  | learning_rate | MindSpore设置学习率的初始值 |
|      | 参数6 |    -   |  decay_steps   | MindSpore进行衰减的step数 |
|      | 参数7 |    -   |  is_stair   | MindSpore `is_stair` 为True时，学习率每 `decay_steps` 衰减一次 |

## 代码示例

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
# out
# [0.1, 0.1, 0.09000000000000001, 0.09000000000000001, 0.08100000000000002, 0.08100000000000002]

# In MindSpore：ExponentialDecayLR
learning_rate = 0.1
decay_rate = 0.9
decay_steps = 4
global_step = ms.Tensor(2, ms.int32)
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
