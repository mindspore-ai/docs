# 比较与torch.optim.RMSProp的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RMSProp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[torch.optim.RMSProp](https://pytorch.org/docs/1.8.0/optim.html#torch.optim.RMSProp)。

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

更多内容详见[mindspore.nn.RMSProp](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.RMSProp.html#mindspore.nn.RMSProp)。

## 差异对比

PyTorch和MindSpore此优化器实现算法不同，详情请参考官网公式。

| 分类 | 子类  | PyTorch      | MindSpore     | 差异                                                 |
| ---- |-----|--------------|---------------|----------------------------------------------------|
| 参数 | 参数1 | params       | params        | 功能一致                                               |
|      | 参数2 | lr           | learning_rate | 功能一致，参数名及默认值不同                                     |
|      | 参数3 | alpha        | decay             | 功能一致，参数名及默认值不同                                     |
|      | 参数4 | eps          | epsilon             | 功能一致，参数名及默认值不同                                     |
|      | 参数5 | weight_decay | weight_decay             | 功能一致                                               |
|      | 参数6 | momentum     | momentum             | 功能一致                                               |
|      | 参数7 | centered     | centered             | 功能一致                                               |
|      | 参数8 | -            | use_locking             | MindSpore的 `use_locking` 用于控制是否更新网络权重，PyTorch无此参数 |
|      | 参数9 | -            | loss_scale             | MindSpore的 `loss_scale` 为梯度缩放系数，PyTorch无此参数       |

### 代码示例

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
