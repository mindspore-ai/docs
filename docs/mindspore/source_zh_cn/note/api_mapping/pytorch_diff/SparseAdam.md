# 比较与torch.optim.SparseAdam的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SparseAdam.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.optim.SparseAdam

```python
class torch.optim.SparseAdam(
    params,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08
)
```

更多内容详见[torch.optim.SparseAdam](https://pytorch.org/docs/1.8.0/optim.html#torch.optim.SparseAdam)。

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

更多内容详见[mindspore.nn.LazyAdam](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.LazyAdam.html#mindspore.nn.LazyAdam)。

## 差异对比

`torch.optim.SparseAdam` 为PyTorch中专门用于稀疏场景的Adam算法；

`mindspore.nn.LazyAdam` 既可以用于常规场景，也可以用于稀疏场景：

- 当输入梯度为稀疏Tensor时，默认参数下 `mindspore.nn.LazyAdam` 与 `torch.optim.SparseAdam` 一致，但 `mindspore.nn.LazyAdam` 当前仅支持CPU后端；

- 当输入梯度为非稀疏时，`mindspore.nn.LazyAdam` 自动执行 `mindspore.nn.Adam` 算法，且支持CPU/GPU/Ascend后端。

| 分类 | 子类  | PyTorch | MindSpore     | 差异                                                 |
| ---- |-----|---------|---------------|----------------------------------------------------|
| 参数 | 参数1 | params  | params        | 功能一致                                               |
|      | 参数2 | lr      | learning_rate | 功能一致，参数名不同                                         |
|      | 参数3 | betas   | beta1, beta2  | 功能一致，参数名不同                                         |
|      | 参数4 | eps     | eps           | 功能一致                                               |
|      | 参数5 | -       | use_locking   | MindSpore的 `use_locking` 表示是否对参数更新加锁保护，PyTorch的无此参数 |
|      | 参数6 | -       | use_nesterov  | MindSpore的 `use_nesterov` 是否使用NAG算法更新梯度，PyTorch的无此参数     |
|      | 参数7 | -       | weight_decay  | PyTorch无此参数                                        |
|      | 参数8 | -       | loss_scale    | MindSpore的 `loss_scale` 为梯度缩放系数，PyTorch的无此参数       |

### 代码示例

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
