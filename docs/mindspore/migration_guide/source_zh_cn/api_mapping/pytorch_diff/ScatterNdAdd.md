# 比较与torch.Tensor.scatter_add_的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/ScatterNdAdd.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.Tensor.scatter_add_

```python
torch.Tensor.scatter_add_(
    dim,
    index,
    src
)
```

更多内容详见[torch.Tensor.scatter_add_](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.scatter_add_)。

## mindspore.ops.ScatterNdAdd

```python
class mindspore.ops.ScatterNdAdd(use_locking=False)(
    input_x,
    indices,
    update
)
```

更多内容详见[mindspore.ops.ScatterNdAdd](https://mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ScatterNdAdd.html#mindspore.ops.ScatterNdAdd)。

## 使用方式

PyTorch：给定输入tensor，更新tensor和索引tensor；将更新tensor按照索引tensor在指定的轴上加到输入tensor上。

MindSpore：给定输入tensor，更新tensor和索引tensor；将更新tensor按照索引tensor加到输入tensor上；
不支持通过参数自定义轴，但可通过调整索引tensor的形状来明确轴。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, no parameter for specifying dimension.
input_x = mindspore.Parameter(Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mindspore.float32), name="x")
indices = Tensor(np.array([[2], [4], [1], [7]]), mindspore.int32)
updates = Tensor(np.array([6, 7, 8, 9]), mindspore.float32)
scatter_nd_add = ops.ScatterNdAdd()
output = scatter_nd_add(input_x, indices, updates)
print(output)
# Out：
# [1. 10. 9. 4. 12. 6. 7. 17.]

# In torch, parameter dim can be set to specify dimension.
input_x = torch.tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float32))
indices = torch.tensor(np.array([2, 4, 1, 7]).astype(np.int64))
updates = torch.tensor(np.array([6, 7, 8, 9]).astype(np.float32))
output = input_x.scatter_add_(dim=0, index=indices, src=updates)
print(output)
# Out:
# tensor([1., 10., 9., 4., 12., 6., 7., 17.])
```
