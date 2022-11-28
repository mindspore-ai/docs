# 比较与torch.bitwise_xor的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/bitwise_xor.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.bitwise_xor

```text
torch.bitwise_xor(input, other, *, out=None) -> Tensor
```

更多内容详见 [torch.bitwise_xor](https://pytorch.org/docs/1.8.1/generated/torch.bitwise_xor.html)。

## mindspore.ops.bitwise_xor

```text
mindspore.ops.bitwise_xor(x, y) -> Tensor
```

更多内容详见 [mindspore.ops.bitwise_xor](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.bitwise_xor.html)。

## 差异对比

PyTorch：如果输入的两tensor数据类型为布尔类型则计算两tensor数据的逻辑异或，否则计算两tensor数据的按位异或。

MindSpore: MindSpore此API实现功能与PyTorch一致，但MindSpore不支持布尔类型的tensor数据。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同 |
|      | 参数2 | other   | y         | 功能一致，参数名不同 |
|      | 参数3 | out     | -         | 不涉及               |

### 代码示例1

两API实现功能一致，用法相同。

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.array([0, 0, 1, -1, 1, 1, 1]), dtype=torch.int32)
other = torch.tensor(np.array([0, 1, 1, -1, -1, 2, 3]), dtype=torch.int32)
output = torch.bitwise_xor(input, other).numpy()
print(output)
# [ 0  1  0  0 -2  3  2]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int32)
y = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int32)
output = ops.bitwise_xor(x, y)
print(output)
# [ 0  1  0  0 -2  3  2]
```
