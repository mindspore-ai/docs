# Differences with torch.optim.RMSProp

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RMSProp.md)

## torch.optim.RMSProp

```python
class torch.optim.RMSProp(
    params,
    lr=0.01,
    alpha=0.99,
    eps=1e-08,
    weight_decay=0,
    momentum=0,
    centered=False
)
```

For more information, see [torch.optim.RMSProp](https://pytorch.org/docs/1.8.0/optim.html#torch.optim.RMSProp).

## mindspore.nn.RMSProp

```python
class mindspore.nn.RMSProp(
    params,
    learning_rate=0.1,
    decay=0.9,
    momentum=0.0,
    epsilon=1e-10,
    use_locking=False,
    centered=False,
    loss_scale=1.0,
    weight_decay=0.0
)
```

For more information, see [mindspore.nn.RMSProp](https://mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.RMSProp.html#mindspore.nn.RMSProp).

## Differences

PyTorch and MindSpore implement different algorithms for this optimizer. Please refer to the formula on the official website for details.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | params       | params        | Consistent function           |
|      | Parameter 2 | lr           | learning_rate | Consistent function, different parameter names and default values                                     |
|      | Parameter 3 | alpha        | decay             | Consistent function, different parameter names and default values                                     |
|      | Parameter 4 | eps          | epsilon             | Consistent function, different parameter names and default values                                     |
|      | Parameter 5 | weight_decay | weight_decay             | Consistent function                                               |
|      | Parameter 6 | momentum     | momentum             | Consistent function                                               |
|      | Parameter 7 | centered     | centered             | Consistent function                                               |
|      | Parameter 8 | -            | use_locking             | MindSpore `use_locking` controls whether to update the network weights, and PyTorch does not have this parameter|
|      | Parameter 9 | -            | loss_scale             | MindSpore `loss_scale` is the gradient scaling factor, and PyTorch does not have this parameter       |

### Code Example

```python
# MindSpore
import mindspore
from mindspore import nn

net = nn.Dense(2, 3)
optimizer = nn.RMSProp(net.trainable_params())
criterion = nn.MAELoss(reduction="mean")

def forward_fn(data, label):
    logits = net(data)
    loss = criterion(logits, label)
    return loss, logits

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

# PyTorch
import torch

model = torch.nn.Linear(2, 3)
criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.RMSProp(model.parameters())
def train_step(data, label):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
```
