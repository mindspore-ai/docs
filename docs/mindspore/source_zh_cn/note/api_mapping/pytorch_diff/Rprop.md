# 比较与torch.optim.Rprop的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Rprop.md)

## torch.optim.Rprop

```python
class torch.optim.Rprop(
    params,
    lr=0.01,
    etas=(0.5, 1.2),
    step_sizes=(1e-06, 50)
)
```

更多内容详见[torch.optim.Rprop](https://pytorch.org/docs/1.8.0/optim.html#torch.optim.Rprop)。

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

更多内容详见[mindspore.nn.Rprop](https://mindspore.cn/docs/zh-CN/r2.1/api_python/nn/mindspore.nn.Rprop.html#mindspore.nn.Rprop)。

## 差异对比

PyTorch和MindSpore此优化器实现算法不同，详情请参考官网公式。

| 分类 | 子类  | PyTorch    | MindSpore     | 差异                              |
| ---- |-----|------------|---------------|---------------------------------|
| 参数 | 参数1 | params     | params        | 功能一致                            |
|      | 参数2 | lr         | learning_rate | 功能一致，参数名及默认值不同                  |
|      | 参数3 | etas       | etas          | 功能一致，参数名不同                      |
|      | 参数4 | step_sizes | step_sizes    | 功能一致                            |
|      | 参数5 | -          | weight_decay  | PyTorch无此参数                 |

### 代码示例

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
