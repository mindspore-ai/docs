# 比较与torch.optim.lr_scheduler.ExponentialLR的功能差异

## torch.optim.lr_scheduler.ExponentialLR

```python
torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma,
    last_epoch=-1,
    verbose=False)
)
```

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

## mindspore.nn.ExponentialDecayLR

```python
mindspore.nn.ExponentialDecayLR(
  learning_rate,
  decay_rate,
  decay_steps,
  is_stair=False
)
```

## 使用方式

PyTorch: 计算方式为lr*gamma^{epoch}。使用时，优化器作为输入，通过调用`step`方法进行学习率的更新。

MindSpore：计算方式为lr*decay_rate^{p}，这种动态学习率的调整方式在mindspore里有两种实现方式：`exponential_decay_lr`预生成学习率列表，将列表传入优化器；`ExponentialDecayLR`则是通过计算图的方式传入优化器中参与训练。

## 代码示例

```python
# In MindSpore：
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
global_step = Tensor(2, mstype.int32)
exponential_decay_lr = nn.ExponentialDecayLR(learning_rate, decay_rate, decay_steps)
result = exponential_decay_lr(global_step)
print(result)
#  out
# 0.09486833

# In torch:
from torch import optim

optim_sgd = optim.SGD(net.parameters(), lr=0.01)
exponential_decay_lr = optim.lr_scheduler.ExponentialLR(optim_sgd, decay_rate, gamma=0.9)
```
