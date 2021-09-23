# 比较与torch.topk的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/TopK.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.topk

```python
torch.topk(
    input,
    k,
    dim=None,
    largest=True,
    sorted=True,
    *,
    out=None
)
```

更多内容详见[torch.topk](https://pytorch.org/docs/1.5.0/torch.html#torch.topk)。

## mindspore.ops.TopK

```python
class mindspore.ops.TopK(
    sorted=False
)(input_x, k)
```

更多内容详见[mindspore.ops.TopK](https://mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.TopK.html#mindspore.ops.TopK)。

## 使用方式

PyTorch: 支持获取指定维度的前k项最大或最小值。

MindSpore：目前仅支持获取最后维度的前k项最大值。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch

# In MindSpore, obtain the first k largest entries of the last dimension.
topk = ops.TopK()
k = 3
input_x = Tensor([[1, 2, 3, 4], [2, 4, 6, 8]], mindspore.float16)
values, indices = topk(input_x, k)
print(values)
print(indices)

# Out：
# [[4. 3. 2.]]
# [[8. 6. 4.]]
# [[3 2 1]]
# [[3 2 1]]

# In torch, obtain the first k largest or smallest entries of a specific dimension.
# largest=True
input_x = torch.tensor([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=torch.float)
dim = 1
output = torch.topk(input_x, k, dim=dim, largest=True)
print(output)

# Out：
# torch.return_types.topk(
# values=tensor([[4., 3., 2.],
#                [8., 6., 4.]]),
# indices=tensor([[3, 2, 1],
#                 [3, 2, 1]]))

# largest=False
output = torch.topk(input_x, k, dim=dim, largest=False)
print(output)

# Out：
# torch.return_types.topk(
# values=tensor([[1., 2., 3.],
#                [2., 4., 6.]]),
# indices=tensor([[0, 1, 2],
#                 [0, 1, 2]]))
```
