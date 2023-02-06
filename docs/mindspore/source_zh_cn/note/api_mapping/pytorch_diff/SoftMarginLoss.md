# 比较与torch.nn.functional.soft_margin_loss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SoftMarginLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.nn.functional.soft_margin_loss

```text
torch.nn.functional.soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean')  -> Tensor/Scalar
```

更多内容详见[torch.nn.functional.soft_margin_loss](https://pytorch.org/docs/1.8.1/nn.functional.html#soft-margin-loss)。

## mindspore.nn.SoftMarginLoss

```text
class mindspore.nn.SoftMarginLoss(reduction='mean')(logits, labels)  -> Tensor/Scalar
```

更多内容详见[mindspore.nn.SoftMarginLoss](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.SoftMarginLoss.html)。

## 差异对比

PyTorch：针对二分类问题的损失函数，用于计算输入Tensor x 和目标值Tensor y （包含1或-1）的二分类损失值。

MindSpore：除两个在PyTorch已弃用的参数不同外，功能上无差异。

| 分类 | 子类  | PyTorch      | MindSpore | 差异                                                         |
| ---- | ----- | ------------ | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 | input | logits | 功能一致，参数名不同|
| | 参数2 | target | labels | 功能一致，参数名不同|
| | 参数3 | size_average | -         | 已弃用，被reduction取代，MindSpore无此参数 |
| | 参数4 | reduce | - | 已弃用，被reduction取代，MindSpore无此参数 |
| | 参数5 | reduction | reduction | - |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor
import torch.nn as nn

logits = torch.FloatTensor([[0.3, 0.7], [0.5, 0.5]])
labels = torch.FloatTensor([[-1, 1], [1, -1]])
output = torch.nn.functional.soft_margin_loss(logits, labels)
print(output.numpy())
# 0.6764238

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

loss = mindspore.nn.SoftMarginLoss()
logits = Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]), mindspore.float32)
labels = Tensor(np.array([[-1, 1], [1, -1]]), mindspore.float32)
output = loss(logits, labels)
print(output)
# 0.6764238
```
