# Differences with torch.optim.Rprop

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Rprop.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png"></a>

## torch.optim.Rprop

```python
class torch.optim.Rprop(
    params,
    lr=0.01,
    etas=(0.5, 1.2),
    step_sizes=(1e-06, 50)
)
```

For more information, see [torch.optim.Rprop](https://pytorch.org/docs/1.8.0/optim.html#torch.optim.Rprop).

## mindspore.nn.Rprop

```python
class mindspore.nn.Rprop(
    params,
    learning_rate=0.1,
    etas=(0.5, 1.2),
    step_sizes=(1e-06, 50),
    weight_decay=0.0,
)
```

For more information, see [mindspore.nn.Rprop](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.Rprop.html#mindspore.nn.Rprop).

## Differences

PyTorch and MindSpore implement different algorithms for this optimizer. Please refer to the formula on the official website for details.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | params     | params        | Consistent function                            |
|      | Parameter 2 | lr         | learning_rate | Consistent function, different parameter names and default values                  |
|      | Parameter 3 | etas       | etas          | onsistent function, different parameter names                      |
|      | Parameter 4 | step_sizes | step_sizes    | Consistent function                            |
|      | Parameter 5 | -          | weight_decay  | PyTorch does not have this parameter                 |

### Code Example

```python
# MindSpore.
import mindspore
from mindspore import nn

net = nn.Dense(2, 3)
optimizer = nn.Rprop(net.trainable_params())
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
optimizer = torch.optim.Rprop(model.parameters())
def train_step(data, label):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
```
