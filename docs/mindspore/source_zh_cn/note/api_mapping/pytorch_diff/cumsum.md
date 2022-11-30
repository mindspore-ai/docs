# 比较与torch.cumsum的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/cumsum.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.cumsum

```text
torch.cumsum(input, dim, *, dtype=None, out=None) -> Tensor
```

更多内容详见 [torch.cumsum](https://pytorch.org/docs/1.8.1/generated/torch.cumsum.html)。

## mindspore.ops.cumsum

```text
mindspore.ops.cumsum(x, axis, dtype=None) -> Tensor
```

更多内容详见 [mindspore.ops.cumsum](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cumsum.html)。

## 差异对比

PyTorch：计算输入Tensor在指定轴上的累加和。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，不过参数设定上有所差异。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | x |功能一致，参数名不同 |
| | 参数2 | dim | axis | 功能一致，参数名不同 |
| | 参数3 | dtype | dtype | - |
| | 参数4 | out | - | 不涉及 |

### 代码示例1

> 当输入tensor相同，累加轴为-1时，对tensor最内层从左到右累加，两API实现相同的功能。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np
a = tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
y = torch.cumsum(a, dim=-1)
print(y.numpy())
# [[ 3.  7. 13. 23.]
#  [ 1.  7. 14. 23.]
#  [ 4.  7. 15. 22.]
#  [ 1.  4. 11. 20.]]

# MindSpore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np
x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
y = ops.cumsum(x, -1)
print(y)
# [[ 3.  7. 13. 23.]
#  [ 1.  7. 14. 23.]
#  [ 4.  7. 15. 22.]
#  [ 1.  4. 11. 20.]]
```

### 代码示例2

> 当输入tensor和累加轴相同，torch.cumsum和MindSpore通过参数dtype设定输出y的数据类型为int8，得到相同的结果。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np
a = tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
y = torch.cumsum(a, dim=0, dtype=torch.int8)
print(y.numpy())
# [[ 3  4  6 10]
#  [ 4 10 13 19]
#  [ 8 13 21 26]
#  [ 9 16 28 35]]
print(y.dtype)
# torch.int8

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
x = Tensor([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]], mindspore.float32)
y = ops.cumsum(x, 0, dtype=mindspore.int8)
print(y)
# [[ 3  4  6 10]
#  [ 4 10 13 19]
#  [ 8 13 21 26]
#  [ 9 16 28 35]]
print(y.dtype)
# Int8
```
