# 比较与torch.optim.AdaMax的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/AdaMax.md)

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

更多内容详见[torch.optim.AdaMax](https://pytorch.org/docs/1.8.0/optim.html#torch.optim.AdaMax)。

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

更多内容详见[mindspore.nn.AdaMax](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.AdaMax.html#mindspore.nn.AdaMax)。

## 差异对比

PyTorch和MindSpore此优化器实现算法不同，详情请参考官网公式。

| 分类 | 子类  | PyTorch      | MindSpore     | 差异                                     |
| ---- |-----|--------------|---------------|----------------------------------------|
| 参数 | 参数1 | params       | params        | 功能一致                                   |
|      | 参数2 | lr           | learning_rate | 功能一致，参数名及默认值不同                         |
|      | 参数3 | betas        | beta1, beta2  | 功能一致，参数名不同                             |
|      | 参数4 | eps          | eps           | 功能一致                                   |
|      | 参数5 | weight_decay | weight_decay  | 功能一致                          |
|      | 参数6 | -            | loss_scale    | MindSpore的 `loss_scale` 为梯度缩放系数，PyTorch无此参数 |

### 代码示例

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
