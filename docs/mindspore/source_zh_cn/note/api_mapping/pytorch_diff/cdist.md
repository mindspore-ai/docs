# 比较与torch.cdist的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/cdist.md)

## torch.cdist

```text
torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
```

更多内容详见[torch.cdist](https://pytorch.org/docs/1.8.1/generated/torch.cdist.html)。

## mindspore.ops.cdist

```text
mindspore.ops.cdist(x1, x2, p=2.0)
```

更多内容详见[mindspore.ops.cdist](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.cdist.html)。

## 差异对比

PyTorch：计算两个Tensor每对列向量之间的p-norm距离。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，精度稍有差异。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 |x1 | x1 | - |
| | 参数2 | x2 | x2 | - |
|  | 参数3 | p | p | - |
| | 参数4 | compute_mode | - | torch中指定是否用矩阵乘法计算欧几里得距离，MindSpore中没有该参数 |

### 代码示例1

```python
# PyTorch
import torch
import numpy as np

x =  torch.tensor(np.array([[1.0, 1.0], [2.0, 2.0]]).astype(np.float32))
y =  torch.tensor(np.array([[3.0, 3.0], [3.0, 3.0]]).astype(np.float32))
output = torch.cdist(x, y, 2.0)
print(output)
# tensor([[2.8284, 2.8284],
#         [1.4142, 1.4142]])

# MindSpore
import mindspore.numpy as np
from mindspore import Tensor
from mindspore import ops

x = Tensor(np.array([[1.0, 1.0], [2.0, 2.0]]).astype(np.float32))
y = Tensor(np.array([[3.0, 3.0], [3.0, 3.0]]).astype(np.float32))
output = ops.cdist(x, y, 2.0)
print(output)
# [[2.828427  2.828427 ]
#  [1.4142135 1.4142135]]

```
