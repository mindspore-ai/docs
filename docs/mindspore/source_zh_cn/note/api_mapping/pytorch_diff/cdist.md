# 比较与torch.cdist的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/cdist.md)

## torch.cdist

```text
torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
```

更多内容详见[torch.cdist](https://pytorch.org/docs/1.8.1/generated/torch.cdist.html)。

## mindspore.ops.cdist

```text
mindspore.ops.cdist(x1, x2, p=2.0)
```

更多内容详见[mindspore.ops.cdist](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.cdist.html)。

## 差异对比

MindSpore此API功能与PyTorch基本一致，MindSpore无法指定是否使用矩阵乘的方式计算向量对之间的欧几里得距离。

PyTorch: 当参数 `compute_mode` 为 ``use_mm_for_euclid_dist_if_necessary`` ，且当 `x1` 或 `x2` 的一个batch中的行向量的个数超过25时，使用矩阵乘的方式计算向量对之间的欧几里得距离。当参数 `compute_mode` 为 ``use_mm_for_euclid_dist`` 时，使用矩阵乘的方式计算向量对之间的欧几里得距离。当参数 `compute_mode` 为 ``donot_use_mm_for_euclid_dist`` 时，不会使用矩阵乘的方式计算向量对之间的欧几里得距离。

MindSpore：无参数 `compute_mode` 以指定是否使用矩阵乘的方式计算向量对之间的欧几里得距离。在 ``GPU`` 和 ``CPU`` 上不会使用矩阵乘的方式计算向量对之间的欧几里得距离，在 ``Ascend`` 上，会使用矩阵乘的方式计算向量对之间的欧几里得距离。

| 分类 | 子类 | PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 |x1 | x1 | - |
| | 参数2 | x2 | x2 | - |
|  | 参数3 | p | p | - |
| | 参数4 | compute_mode | - | PyTorch中指定是否用矩阵乘的方式计算欧几里得距离的参数，MindSpore中没有该参数 |

### 代码示例

```python
# PyTorch
import torch
import numpy as np

torch.set_printoptions(precision=7)
x =  torch.tensor(np.array([[1.0, 1.0], [2.0, 2.0]]).astype(np.float32))
y =  torch.tensor(np.array([[3.0, 3.0], [3.0, 3.0]]).astype(np.float32))
output = torch.cdist(x, y, 2.0)
print(output)
# tensor([[2.8284271, 2.8284271],
#         [1.4142135, 1.4142135]])

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
