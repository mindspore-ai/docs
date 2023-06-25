# Function Differences between torch.optim.AdaMax and mindspore.nn.AdaMax

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AdaMax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.optim.AdaMax

```python
class torch.optim.AdaMax(
    params,
    lr=0.002,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0
)
```

For more information, see [torch.optim.AdaMax](https://pytorch.org/docs/1.8.0/optim.html#torch.optim.AdaMax).

## mindspore.nn.AdaMax

```python
class mindspore.nn.AdaMax(
    params,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-08,
    weight_decay=0.0,
    loss_scale=1.0
)
```

For more information, see [mindspore.nn.AdaMax](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.AdaMax.html#mindspore.nn.AdaMax).

## Differences

PyTorch and MindSpore implement different algorithms for this optimizer. Please refer to the formula on the official website for details.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | params       | params        | Consistent function                                   |
|      | Parameter 2 | lr           | learning_rate | Same function, different parameter names and default values                         |
|      | Parameter 3 | betas        | beta1, beta2  | Same function, different parameter names|
|      | Parameter 4 | eps          | eps           | Consistent function                                   |
|      | Parameter 5 | weight_decay | weight_decay  | Consistent function                          |
|      | Parameter 6 | -            | loss_scale    | MindSpore `loss_scale` is the gradient scaling factor, and PyTorch does not have this parameter |

### Code Example

```python
# MindSpore
import mindspore
from mindspore import nn

net = nn.Dense(2, 3)
optimizer = nn.AdaMax(net.trainable_params())
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
optimizer = torch.optim.AdaMax(model.parameters())
def train_step(data, label):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
```