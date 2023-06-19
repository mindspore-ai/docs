# 比较与torch.optim.Adagrad的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Adagrad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[torch.optim.Adagrad](https://pytorch.org/docs/1.8.0/optim.html#torch.optim.Adagrad)。

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

更多内容详见[mindspore.nn.Adagrad](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Adagrad.html#mindspore.nn.Adagrad)。

## 差异对比

PyTorch和MindSpore此优化器实现算法不同，PyTorch在每一轮迭代中对学习率进行衰减，且在除法计算中加入 `eps` 以维持计算稳定性，MindSpore中无此过程，详情请参考官网公式。

| 分类 | 子类  | PyTorch                   | MindSpore     | 差异                                               |
| ---- |-----|---------------------------|---------------|--------------------------------------------------|
| 参数 | 参数1 | params                    | params        | 功能一致                                             |
|      | 参数2 | lr                        | learning_rate | 功能一致，参数名及默认值不同                                   |
|      | 参数3 | lr_decay                  | -             | PyTorch的 `lr_decay` 表示学习率的衰减值，MindSpore无此参数      |
|      | 参数4 | weight_decay              | weight_decay             | 功能一致                                             |
|      | 参数5 | initial_accumulator_value | accum             | 功能一致，参数名及默认值不同                                   |
|      | 参数6 | eps                       | -             | PyTorch的 `eps` 用于加在除法的分母上以增加计算稳定性，MindSpore无此参数  |
|      | 参数7 | -                         | update_slots             | MindSpore的 `update_slots` 表示是否更新累加器，PyTorch无此参数 |
|      | 参数8 | -                         | loss_scale             | MindSpore的 `loss_scale` 为梯度缩放系数，PyTorch无此参数     |

### 代码示例

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
