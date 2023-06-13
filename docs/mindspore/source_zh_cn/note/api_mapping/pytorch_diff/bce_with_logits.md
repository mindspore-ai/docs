# 比较与torch.nn.functional.binary_cross_entropy_with_logits的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/bce_with_logits.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png"></a>

## torch.nn.functional.binary_cross_entropy_with_logits

```text
torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
```

更多内容详见[torch.nn.functional.binary_cross_entropy_with_logits](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.binary_cross_entropy_with_logits)。

## mindspore.ops.binary_cross_entropy_with_logits

```text
mindspore.ops.binary_cross_entropy_with_logits(logits, label, weight, pos_weight, reduction='mean')
```

更多内容详见[mindspore.ops.binary_cross_entropy_with_logits](https://mindspore.cn/docs/zh-CN/r1.11/api_python/ops/mindspore.ops.binary_cross_entropy_with_logits.html)。

## 差异对比

PyTorch：将Sigmoid层和BCELoss组合在一个函数中计算预测值和目标值之间的二值交叉熵损失，使其比分开使用Sigmoid后跟BCELoss在数值上更加稳定。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅输入参数 `weight` 和 `pos_weight` 默认值未设定。

| 分类 | 子类  | PyTorch   | MindSpore | 差异                                                         |
| ---- | ----- | --------- | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 | input     | logits    | 功能一致，参数名不同                                         |
|      | 参数2 | target    | label    | 功能一致，参数名不同                                         |
|      | 参数3 | weight    | weight    | 功能一致，参数名默认值未设定                                      |
|      | 参数4 | size_average    | -    | PyTorch的已弃用参数，功能由reduction参数取代          |
|      | 参数5 | reduce    | -    | PyTorch的已弃用参数，功能由reduction参数取代                 |
|      | 参数6 | reduction | reduction | 功能一致，默认值不同                                    |
|      | 参数7 | pos_weight    | pos_weight    | 功能一致，参数名默认值未设定                                      |

### 代码示例1

> 两API实现功能一致，`weight` 和 `pos_weight` 都设定为1的情况下，MindSpore能得到和PyTorch一样的结果。

```python
import numpy as np
import mindspore
from mindspore import Tensor, ops

logits = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), mindspore.float32)
label = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), mindspore.float32)
weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
pos_weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
output = ops.binary_cross_entropy_with_logits(logits, label, weight, pos_weight)
print(output)
# 0.34636116

import torch

logits = torch.tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]))
label = torch.tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]))
output = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
print(output)
# tensor(0.3464, dtype=torch.float64)
```
