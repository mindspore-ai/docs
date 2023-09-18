# 比较与torch.optim.Adam的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Adam.md)

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

更多内容详见[torch.optim.Adam](https://pytorch.org/docs/1.8.0/optim.html#torch.optim.Adam)。

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

更多内容详见[mindspore.nn.Adam](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.Adam.html#mindspore.nn.Adam)。

## 差异对比

`mindspore.nn.Adam` 可以覆盖 `torch.optim.Adam` 的功能，且默认参数下功能一致。`mindspore.nn.Adam` 中相比PyTorch多出的入参用于控制其他功能，详情请参考官网注释。

| 分类 | 子类  | PyTorch                   | MindSpore     | 差异                                               |
| ---- |-----|---------------------------|---------------|--------------------------------------------------|
| 参数 | 参数1 | params                     | params          | 一致                                             |
|      | 参数2 | lr                        | learning_rate   | 功能一致，参数名不同                                |
|      | 参数3 | eps                       | eps             | 一致                                        |
|      | 参数4 | weight_decay              | weight_decay    | 一致                                             |
|      | 参数5 | amsgrad                   | use_amsgrad     | 功能一致，参数名不同                                  |
|      | 参数6 | betas                     | beta1, beta2    | 功能一致，参数名不同  |
|      | 参数7 | -                         | use_locking     | MindSpore的 `use_locking` 表示是否更新累加器，PyTorch无此参数 |
|      | 参数8 | -                         | use_nesterov    | MindSpore的 `use_nesterov` 表示是否使用NAG算法更新梯度，PyTorch无此参数     |
|      | 参数9 | -                         | loss_scale      | MindSpore的 `loss_scale` 为梯度缩放系数，PyTorch无此参数     |
|      | 参数10 | -                        | kwargs          | MindSpore中 `kwargs` 中传入的入参名为"use_lazy"和"use_offload"的参数可被解析生效，表示是否使用Lazy Adam算法或Offload Adam算法，PyTorch无此参数     |

### 代码示例

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
