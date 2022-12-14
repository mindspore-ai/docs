# 比较与torch.index_select的功能差异

## torch.index_select

```text
torch.index_select(input, dim,index) -> Tensor
```

更多内容详见 [torch.index_select](https://pytorch.org/docs/1.8.1/generated/torch.index_select.html#torch.index_select)。

## mindspore.ops.gather

```text
mindspore.ops.gather(input_params, input_indices, axis) -> Tensor
```

更多内容详见 [mindspore.ops.gather](https://www.mindspore.cn/docs/zh-CN/r1.10/api_python/ops/mindspore.ops.gather.html)。

## 差异对比

PyTorch dim 上 index 索引对应的元素组成的切片。

MindSpore：MindSpore此API实现功能与PyTorch一致， 仅参数名不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | input_params |功能一致， 参数名不同 |
| | 参数2 | dim | axis | 功能一致， 参数名不同|
| | 参数3 | index | input_indices |功能一致， 参数名不同 |

### 代码示例

说明：两API实功能一致， 用法相同。

```python
# PyTorch
import torch
form torch import tensor

input = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float32)
indices = tensor([0, 2], dtype=torch.int32)
dim = 1
out = torch.index_select(input, dim, indices).numpy()
print(out)
# [[ 1.  3.]
#  [ 5.  7.]
#  [ 9. 11.]]

# MindSpore
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor

input_params = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), mindspore.float32)
input_indices = Tensor(np.array([0, 2]), mindspore.int32)
axis = 1
output = ops.gather(input_params, input_indices, axis)
print(output)
# [[ 1.  3.]
#  [ 5.  7.]
#  [ 9. 11.]]
```
