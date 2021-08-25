# The differences of inputs setting with torch.optim.optimizer

## torch.optim.optimizer()

```python
class torch.optim.Optimizer(
    params,
    defaults
)
```

## mindspore.nn.Optimizer()

```python
class mindspore.nn.Optimizer(
    learning_rate,
    parameters,
    weight_decay=0.0,
    loss_scale=1.0
)
```

## Differences

### parameters setting

- **Default function interface**

MindSpore： `params` can be passed by interface `trainable_params`.

```python
from mindspore import nn

net = Net()
optim_sgd = nn.SGD(params=net.trainable_params())
```

PyTorch： `params` can be passed by interface `parameters`.

```python
from torch import optim

net = Net()
optim_sgd = optim.SGD(params=net.parameters(), lr=0.01)
```

- **User-defined input parameters**

MindSpore：Firstly, get all the parameters in the network by `get_parameters` method, then filter parameters under certain conditions, like names of them, and pass it to the optimizer.

```python
from mindspore import nn

net = Net()
all_params = net.get_parameters()
no_conv_params = list(filter(lambda x: "conv" not in x.name, all_params))
optim_sgd = nn.SGD(no_conv_params)
```

PyTorch：Firstly, get all the parameters in the network by `named_parameters` method, then filter parameters under certain conditions, like names of them, and pass it to the optimizer.

```python
from torch import optim

net = Net()
all_params = net.named_parameters()
no_conv_params = []
for pname, p in all_params:
  if "conv" not in pname:
    no_conv_params.append(p)
optim_sgd = optim.SGD(no_conv_params, lr=0.01)
```

## learning_rate setting

fix learning rate：same.

dynamic learning rate:

- mindspore：

```python
from mindspore import nn
from mindspore import Tensor
from mindspore import dtype as mstype

# dynamic_lr
milestone = [2, 5, 10]
learning_rates = [0.1, 0.05, 0.01]
lr_dynamic = nn.dynamic_lr.piecewise_constant_lr(milestone, learning_rates)
print(lr_dynamic)

# learning_rate_schedule
lr_schedule = nn.learning_rate_schedule.PolynomialDecayLR(learning_rate=0.1,
                                   end_learning_rate=0.01,
                                   decay_steps=4,
                                   power=0.5 )

global_step = Tensor(2, mstype.int32)
result = lr_schedule(global_step)
print(result)

# lr as input of optimizer
optimizer1 = nn.Momentum(net.trainable_params(), learning_rate=lr_dynamic, momentum=0.9, weight_decay=0.9)
optimizer2 = nn.Momentum(net.trainable_params(), learning_rate=lr_schedule, momentum=0.9, weight_decay=0.9)
```

```python
[0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01]
0.0736396
```

- pytorch：

```python
from torch import optim

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.ExponentialLR(optimizer, gamma=0.9)

for epoch in range(20):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

## weight_decay setting

Same.

## loss_scale setting

- mindspore：As an input parameter of the optimizer, it should be used with FixedLossScaleManager.
- pytorch：Do not set the mixed precision separately for the optimizer.

## Groups of parameters

Both PyTorch and MindSpore support grouping parameters and can be used in similar ways. But mindspore only support groups  for 'params'，'weight_decay'，'lr'，'grad_centralizaiton'; pytorch support groups for all optimizer inputs.

>Currently, there are individual optimizers in Mindspore and pytorch that do not support grouping parameters. For details, refer to the instructions of each optimizer.

MindSpore：

```python
from mindspore import nn

net = Net()

conv_params = list(filter(=lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: "conv" not in x.name, net.trainable_params()))

fix_lr = 0.01
polynomial_decay_lr = nn.learning_rate_schedule.PolynomialDecayLR(learning_rate=0.1,
                                   end_learning_rate=0.01,
                                   decay_steps=4,
                                   power=0.5 )

group_params = [{'params': conv_params, 'weight_decay': 0.01, 'lr': fix_lr},
                {'params': no_conv_params, 'lr': polynomial_decay_lr},
                {'order_params': net.trainable_params()}]

optim_sgd = nn.SGD(group_params, learning_rate=0.1)
```

PyTorch:

```python
from torch import optim

net = Net()

all_params = net.parameters()
conv_params = []
no_conv_params = []

for pname, p in net.named_parameters():
  if 'conv' in pname:
    conv_params += [p]
  else:
    no_conv_params += [p]

group_params = [{'params': conv_params, 'weight_decay': 0.01, 'lr': fix_lr},
                    {'params': no_conv_params, 'nesterov'=True}]

optim_sgd = optim.SGD(group_params, lr=0.01)
```
