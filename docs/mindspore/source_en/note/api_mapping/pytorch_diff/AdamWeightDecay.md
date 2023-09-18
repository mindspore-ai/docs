# Differences between torch.optim.AdamW and mindspore.nn.AdamWeightDecay

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/AdamWeightDecay.md)

## torch.optim.AdamW

```python
class torch.optim.AdamW(
    params,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.01,
    amsgrad=False
)
```

For more information, see [torch.optim.AdamW](https://pytorch.org/docs/1.8.1/optim.html#torch.optim.AdamW).

## mindspore.nn.AdamWeightDecay

```python
class mindspore.nn.AdamWeightDecay(
    params,
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-6,
    weight_decay=0.0
)
```

For more information, see [mindspore.nn.AdamWeightDecay](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.AdamWeightDecay.html#mindspore.nn.AdamWeightDecay).

## Differences

The code implementation and parameter update logic of `mindspore.nn.AdamWeightDecay` optimizer is different from `torch.optim.AdamW`. For more information, please refer to the docs of official website.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | params       | params        | Consistent function                                            |
|      | Parameter 2 | lr           | learning_rate | Same function, different parameter names and default values                                  |
|      | Parameter 3 | betas        | beta1, beta2  | Same function, different parameter names            |
|      | Parameter 4 | eps          | eps           | Same function, different default values             |
|      | Parameter 5 | weight_decay | weight_decay  | Consistent function                     |
|      | Parameter 6 | amsgrad      | -             | PyTorch `amsgrad` indicates whether to apply the amsgrad algorithm, and MindSpore does not have this parameter |

## Code Example

```python
# MindSpore
import mindspore
from mindspore import nn

net = nn.Dense(2, 3)
optimizer = nn.AdamWeightDecay(net.trainable_params())
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
optimizer = torch.optim.AdamW(model.parameters())
def train_step(data, label):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
```