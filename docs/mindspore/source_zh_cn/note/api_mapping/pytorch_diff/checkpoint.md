# 比较与torch.utils.checkpoint.checkpoint的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/checkpoint.md)

## torch.utils.checkpoint.checkpoint

```text
torch.utils.checkpoint.checkpoint(function, preserve_rng_state=True, *args)
```

更多内容详见[torch.utils.checkpoint.checkpoint](https://pytorch.org/docs/1.8.1/checkpoint.html#torch.utils.checkpoint.checkpoint)。

## mindspore.nn.Cell.recompute

```text
mindspore.nn.Cell.recompute(mp_comm_recompute=True, parallel_optimizer_comm_recompute=False)
```

更多内容详见[mindspore.nn.Cell.recompute](https://www.mindspore.cn/docs/zh-CN/br_base/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.recompute)。

## 差异对比

PyTorch：是torch.utils.checkpoint类的一个方法，对指定的Cell进行包装，添加一个中间层，前向计算时使用torch.no_grad进行计算，反向传播时重新计算所需中间值。

MindSpore：定义为cell或者primitive的一个方法，调用时直接使用指定的cell.recompute()或primitive.recompute()来指定重计算。

| 分类 | 子类   | PyTorch | MindSpore  | 差异 |
| ---- | ------ | -------| -----------| ------|
| 参数 | 参数1  | function | mp_comm_recompute  | function表示需要包装的Cell，mp_comm_recompute表示在自动并行或半自动并行模式下，指定Cell内部由模型并行引入的通信操作是否重计算 |
|      | 参数2  | preserve_rng_state | parallel_optimizer_comm_recompute | preserve_rng_state表示是否保存随机数生成器的状态，parallel_optimizer_comm_recompute表示在自动并行或半自动并行模式下，指定Cell内部由优化器并行引入的AllGather通信是否重计算 |
|      | 参数3  | *args |  | 表示function函数的入参 |

### 代码示例1

```python
# PyTorch
import torch
from torch.utils.checkpoint import checkpoint as cp
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 2)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = cp(self._checkpoint, x)
        x = self.fc3(x)
        return x

    def _checkpoint(self, x):
        x = nn.functional.relu(self.fc2(x))
        return x

x = torch.randn(2, 12)
model = Net()
output = model(x)
output.sum().backward()
print(model.fc1.weight.grad)
```

```python
# mindspore
import mindspore
import numpy as np
from mindspore import nn, ops

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(12, 24)
        self.relu1 = ops.ReLU()
        self.fc2 = nn.Dense(24, 48)
        self.fc2.recompute()
        self.relu2 = ops.ReLU()
        self.relu2.recompute()
        self.fc3 = nn.Dense(48, 2)

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = Net()

def forward_fn(data, label):
    logits = model(data)
    loss = logits.sum()
    return loss, logits

grad_fn = mindspore.value_and_grad(forward_fn, None, (model.fc1.weight,), has_aux=True)

x = mindspore.Tensor(np.random.randn(2, 12), dtype=mindspore.float32)
label = mindspore.Tensor(np.random.randn(2, 12), dtype=mindspore.float32)
(loss, _), grads = grad_fn(x, label)
print(grads)
```
