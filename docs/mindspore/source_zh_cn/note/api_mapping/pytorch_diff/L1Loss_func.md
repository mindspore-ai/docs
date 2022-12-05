# 比较与torch.nn.functional.l1_loss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/L1Loss_func.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.functional.l1_loss

``` text
torch.nn.functional.l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor
```

更多内容详见[torch.nn.functional.l1_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#l1-loss)。

## mindspore.nn.L1Loss

```text
mindspore.nn.L1Loss(reduction='mean')(logits, labels) -> Tensor
```

更多内容详见[MindSpore.nn.L1Loss](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.L1Loss.html)。

## 差异对比

PyTorch：functional.l1_loss等价于L1Loss。

MindSpore：包含PyTorch功能，当logits和labels的shape不同但可以互相传播时，仍可运行，但PyTorch不可以。

| 分类 | 子类  | PyTorch   | MindSpore | 差异                 |
| ---- | ----- | --------- | --------- | -------------------- |
| 参数  | 参数1| input     | logits    | 功能一致，参数名不同 |
|      | 参数2| target    | labels    | 功能一致，参数名不同 |
|      | 参数3 | size_average     | -    | 已弃用，功能由reduction接替 |
|      | 参数4 | reduce    | -    | 已弃用，功能由reduction接替|
|      | 参数5 | reduction | reduction | - |

### 代码示例1

> 两API功能一致，用法相同。

```python
# PyTorch
import torch
import torch.nn as nn

loss = nn.functional.l1_loss
input = torch.tensor([2,2,3], dtype=torch.float32)
target = torch.tensor([1,2,2], dtype=torch.float32)
output = loss(input, target)
output = output.detach().numpy()
print(output)
# 0.6666667

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np

loss = nn.L1Loss()
logits = Tensor(np.array([2, 2, 3]), mindspore.float32)
labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
output = loss(logits, labels)
print(output)
# 0.6666667
```
