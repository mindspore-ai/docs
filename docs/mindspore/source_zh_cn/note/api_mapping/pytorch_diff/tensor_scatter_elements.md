# 比较与torch.Tensor.scatter_的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/tensor_scatter_elements.md)

## torch.Tensor.scatter_

```text
torch.Tensor.scatter_(dim, index, src, reduce) -> Tensor
```

更多内容详见[torch.Tensor.scatter_](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.scatter_)。

## mindspore.ops.tensor_scatter_elements

```text
mindspore.ops.tensor_scatter_elements(
    input_x,
    indices,
    updates,
    axis=0,
    reduction='none'
) -> Tensor
```

更多内容详见[mindspore.ops.tensor_scatter_elements](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.tensor_scatter_elements.html)。

## 差异对比

PyTorch：用给定的值替换Tensor中指定索引位置的元素。

MindSpore：MindSpore此算子实现功能与PyTorch一致，PyTorch中该接口为Tensor接口，调用方式略有不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | dim | axis | 功能一致，参数名不同 |
|    | 参数2 | index | indices | 功能一致，参数名不同 |
|    | 参数3 | src | updates | 功能一致，参数名不同 |
|    | 参数4 | reduce | reduction | 规约计算方式，目前MindSpore仅支持“none”和“add”模式 |
|    | 参数5 | - | input_x | PyTorch中该接口为Tensor接口 |

### 代码示例

> 两API实现功能一致。

```python
# PyTorch
import torch

t = torch.zeros((3, 4), dtype=torch.float32)
indices = torch.tensor([[1, 2], [0, 1]])
values = torch.tensor([[3, 4], [5, 6]], dtype=torch.float32)
t.scatter_(0, indices, values)
print(t)
# tensor([[5., 0., 0., 0.],
#         [3., 6., 0., 0.],
#         [0., 4., 0., 0.]])

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor, Parameter
from mindspore import ops

input_x = Parameter(Tensor(np.zeros((3, 4)), mindspore.float32), name="x")
indices = Tensor(np.array([[1, 2], [0, 1]]), mindspore.int32)
updates = Tensor(np.array([[3, 4], [5, 6]]), mindspore.float32)
axis = 0
reduction = "none"
output = ops.tensor_scatter_elements(input_x, indices, updates, axis, reduction)
print(output)
# [[5. 0. 0. 0.]
#  [3. 6. 0. 0.]
#  [0. 4. 0. 0.]]
```
