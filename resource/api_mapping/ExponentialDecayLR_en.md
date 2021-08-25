# Comparing the function difference with torch.optim.lr_scheduler.ExponentialLR

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

## Differences

PyTorch: The function of calculating the learning rate for each step is lr*gamma^{epoch}. In the training stage, the optimizer should be passed into the lr scheduler, then the step method will be implemented.

MindSpore: The function of calculating learning rate for each step is lr*decay_rate^{p}, which is implemented in two ways in MindSpore: `exponential_decay_lr` pregenerates a list of learning rates and passes the list to the optimizer; secondly, `ExponentialDecayLR` instance is passed into the optimizer, during the training process, the optimizer calls the instance taking the current step as the input to get the current learning rate.

## Code Example

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
