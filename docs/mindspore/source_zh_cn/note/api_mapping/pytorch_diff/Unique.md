# 比较与torch.unique的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Unique.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## torch.unique

```python
torch.unique(
    input,
    sorted=True,
    return_inverse=False,
    return_counts=False,
    dim=None
)
```

更多内容详见[torch.unique](https://pytorch.org/docs/1.5.0/torch.html#torch.unique)。

## mindspore.ops.Unique

```python
class mindspore.ops.Unique(*args, **kwargs)(x)
```

更多内容详见[mindspore.ops.Unique](https://mindspore.cn/docs/zh-CN/r1.7/api_python/ops/mindspore.ops.Unique.html#mindspore.ops.Unique)。

## 使用方式

PyTorch：可通过设置参数来确定输出是否排序，是否输出输入的tensor的各元素在输出tensor中的位置索引，是否输出各唯一值在输入的tensor中的数量。

MindSpore：升序输出所有的唯一值，以及输入的tensor的各元素在输出tensor中的位置索引。

## 代码示例

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, the tensor containing unique elements in ascending order.
# As well as another tensor containing the corresponding indices will be directly returned.
x = Tensor(np.array([1, 2, 5, 2]), mindspore.int32)
unique = ops.Unique()
output, indices = unique(x)
print(output)
print(indices)
# Out：
# [1 2 5]
# [0 1 2 1]

# In torch, parameters can be set to determine whether to output tensor containing unique elements in ascending order.
# As well as whether to output tensor containing corresponding indices.
x = torch.tensor([1, 2, 5, 2])
output, indices = torch.unique(x, sorted=True, return_inverse=True)
print(output)
print(indices)
# Out：
# tensor([1, 2, 5])
# tensor([0, 1, 2, 1])
```