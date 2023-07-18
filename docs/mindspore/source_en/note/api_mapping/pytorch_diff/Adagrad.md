# Differences with torch.optim.Adagrad

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Adagrad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png"></a>

## torch.optim.Adagrad

```python
class torch.optim.Adagrad(
    params,
    lr=0.01,
    lr_decay=0,
    weight_decay=0,
    initial_accumulator_value=0,
    eps=1e-10
)
```

For more information, see [torch.optim.Adagrad](https://pytorch.org/docs/1.8.1/optim.html#torch.optim.Adagrad).

## mindspore.nn.Adagrad

```python
class mindspore.nn.Adagrad(
    params,
    accum=0.1,
    learning_rate=0.001,
    update_slots=True,
    loss_scale=1.0,
    weight_decay=0.0
)
```

For more information, see [mindspore.nn.Adagrad](https://mindspore.cn/docs/en/r2.1/api_python/nn/mindspore.nn.Adagrad.html#mindspore.nn.Adagrad).

## Differences

PyTorch and MindSpore implement different algorithms for this optimizer. PyTorch decays the learning rate in each round of iteration and adds `eps` to the division calculation to maintain computational stability, while MindSpore does not have this process.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | params                    | params        | Consistent function                                             |
|      | Parameter 2 | lr                        | learning_rate | Same function, different parameter names and default values                                   |
|      | Parameter 3 | lr_decay                  | -             | PyTorch's `lr_decay` indicates the decay value of the learning rate, and MindSpore does not have this parameter      |
|      | Parameter 4 | weight_decay              | weight_decay             | Consistent function                                             |
|      | Parameter 5 | initial_accumulator_value | accum             | Same function, different parameter names and default values                                   |
|      | Parameter 6 | eps                       | -             | PyTorch `eps` is used to add to the denominator of a division to increase computational stability, and MindSpore does not have this parameter  |
|      | Parameter 7 | -                         | update_slots             | MindSpore `update_slots` indicates whether to update the accumulator, and PyTorch does not have this parameter |
|      | Parameter 8 | -                         | loss_scale             | MindSpore `loss_scale` is the gradient scaling factor, and PyTorch does not have this parameter    |

### Code Example

```python
# MindSpore
import mindspore
from mindspore import nn

net = nn.Dense(2, 3)
optimizer = nn.Adagrad(net.trainable_params())
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
optimizer = torch.optim.Adagrad(model.parameters())
def train_step(data, label):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
```