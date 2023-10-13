# Differences with torch.optim.Adam

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Adam.md)

## torch.optim.Adam

```python
class torch.optim.Adam(
    params,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0,
    amsgrad=False
)
```

For more information, see [torch.optim.Adam](https://pytorch.org/docs/1.8.0/optim.html#torch.optim.Adam).

## mindspore.nn.Adam

```python
class mindspore.nn.Adam(
    params,
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    use_locking=False,
    use_nesterov=False,  
    weight_decay=0.0,
    loss_scale=1.0,
    use_amsgrad=False,
    **kwargs
)
```

For more information, see [mindspore.nn.Adam](https://mindspore.cn/docs/en/r2.2/api_python/nn/mindspore.nn.Adam.html#mindspore.nn.Adam).

## Differences

`mindspore.nn.Adam` can override the function of `torch.optim.Adam`, and the function is the same with default parameters. The extra inputs in `mindspore.nn.Adam` compared to PyTorch are used to control other functions. See the notes on the website for details.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | params                     | params          | Consistent                                             |
|      | Parameter 2 | lr                        | learning_rate   | Same function, different parameter names                                |
|      | Parameter 3 | eps                       | eps             | Consistent                                        |
|      | Parameter 4 | weight_decay              | weight_decay    | Consistent                                             |
|      | Parameter 5 | amsgrad                   | use_amsgrad     | Same function, different parameter names                                  |
|      | Parameter 6 | betas                     | beta1, beta2    | Same function, different parameter names  |
|      | Parameter 7 | -                         | use_locking     | MindSpore `use_locking` indicates whether to update the accumulator, and PyTorch does not have this parameter |
|      | Parameter 8 | -                         | use_nesterov    | MindSpore `use_nesterov` indicates whether to update the gradient using the NAG algorithm, and PyTorch does not have this parameter     |
|      | Parameter 9 | -                         | loss_scale      | MindSpore `loss_scale` is the gradient scaling factor, and PyTorch does not have this parameter     |
|      | Parameter 10 | -                        | kwargs          | The parameters "use_lazy" and "use_offload" passed into `kwargs` in MindSpore can be resolved to indicate whether to use the Lazy Adam algorithm or the Offload Adam algorithm, and PyTorch does not have this parameter     |

### Code Example

```python
# MindSpore
import mindspore
from mindspore import nn

net = nn.Dense(2, 3)
optimizer = nn.Adam(net.trainable_params())
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
optimizer = torch.optim.Adam(model.parameters())
def train_step(data, label):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
```
