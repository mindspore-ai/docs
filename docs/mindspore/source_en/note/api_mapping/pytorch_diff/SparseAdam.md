# Differences with torch.optim.SparseAdam

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SparseAdam.md)

## torch.optim.SparseAdam

```python
class torch.optim.SparseAdam(
    params,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08
)
```

For more information, see [torch.optim.SparseAdam](https://pytorch.org/docs/1.8.0/optim.html#torch.optim.SparseAdam).

## mindspore.nn.LazyAdam

```python
class mindspore.nn.LazyAdam(
    params,
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    use_locking=False,
    use_nesterov=False,
    weight_decay=0.0,
    loss_scale=1.0
)
```

For more information, see [mindspore.nn.LazyAdam](https://mindspore.cn/docs/en/r2.3/api_python/nn/mindspore.nn.LazyAdam.html#mindspore.nn.LazyAdam).

## Differences

`torch.optimize.SparseAdam` is an Adam algorithm in PyTorch specifically for sparse scenarios.

`mindspore.nn.LazyAdam` can be used for both regular and sparse scenarios:

- `mindspore.nn.LazyAdam` is consistent with `torch.optimize.SparseAdam` with default parameters when the input gradient is a sparse Tensor, but `mindspore.nn.LazyAdam` currently only supports CPU backends;

- When the input gradient is a non-sparse Tensor, `mindspore.nn.LazyAdam` automatically executes the `mindspore.nn.Adam` algorithm, and supports CPU/GPU/Ascend backends.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | params  | params        | Consistent function                                               |
|      | Parameter 2 | lr      | learning_rate | Same function, different parameter names                                         |
|      | Parameter 3 | betas   | beta1, beta2  | Same function, different parameter names                                         |
|      | Parameter 4 | eps     | eps           | Consistent function                                               |
|      | Parameter 5 | -       | use_locking   | MindSpore `use_locking` indicates whether parameter updates are protected by locking, and PyTorch does not have this parameter |
|      | Parameter 6 | -       | use_nesterov  | Whether MindSpore `use_nesterov` uses the NAG algorithm to update the gradient, and PyTorch does not have this parameter     |
|      | Parameter 7 | -       | weight_decay  | PyTorch does not have this parameter                                        |
|      | Parameter 8 | -       | loss_scale    | MindSpore's `loss_scale` is the gradient scaling factor, and PyTorch does not have this parameter       |

### Code Example

```python
# MindSpore.
import mindspore
from mindspore import nn

net = nn.Dense(2, 3)
optimizer = nn.LazyAdam(net.trainable_params())
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
optimizer = torch.optim.SparseAdam(model.parameters())
def train_step(data, label):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
```
