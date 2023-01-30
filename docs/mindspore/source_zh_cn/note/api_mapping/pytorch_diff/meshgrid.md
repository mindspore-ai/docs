# 比较与torch.meshgrid的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/meshgrid.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.meshgrid

```text
torch.meshgrid(
    *tensors,
    indexing=None)
```

更多内容详见[torch.meshgrid](https://pytorch.org/docs/1.8.1/generated/torch.meshgrid.html)。

## mindspore.ops.meshgrid

```text
mindspore.ops.meshgrid(*inputs, indexing='xy')
```

更多内容详见[mindspore.ops.meshgrid](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.meshgrid.html)。

## 差异对比

PyTorch：从给定的tensors生成网格矩阵。tensors如果是scalar的list，则scalar将自动被视为大小为(1,)的张量。

MindSpore：MindSpore此API实现功能与PyTorch一致。inputs参数只支持Tensor，不支持scalar。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | tensors  | inputs | 功能一致 |
| | 参数2 | indexing | indexing |功能一致，默认值不相同，torch默认值为'ij'，mindspore默认值为'xy' |

### 代码示例1

```python
# PyTorch
import torch

x = torch.tensor(np.array([1, 2, 3, 4]).astype(np.int32))
y = torch.tensor(np.array([5, 6, 7]).astype(np.int32))
z = torch.tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
inputs = (x, y, z)
output = torch.meshgrid(inputs, indexing='xy')
print(output)
#(tensor([[[1, 1, 1, 1, 1],
#         [2, 2, 2, 2, 2],
#         [3, 3, 3, 3, 3],
#         [4, 4, 4, 4, 4]],

#        [[1, 1, 1, 1, 1],
#         [2, 2, 2, 2, 2],
#         [3, 3, 3, 3, 3],
#         [4, 4, 4, 4, 4]],

#        [[1, 1, 1, 1, 1],
#         [2, 2, 2, 2, 2],
#         [3, 3, 3, 3, 3],
#         [4, 4, 4, 4, 4]]], dtype=torch.int32), tensor([[[5, 5, 5, 5, 5],
#         [5, 5, 5, 5, 5],
#         [5, 5, 5, 5, 5],
#         [5, 5, 5, 5, 5]],

#        [[6, 6, 6, 6, 6],
#         [6, 6, 6, 6, 6],
#         [6, 6, 6, 6, 6],
#         [6, 6, 6, 6, 6]],

#        [[7, 7, 7, 7, 7],
#         [7, 7, 7, 7, 7],
#         [7, 7, 7, 7, 7],
#         [7, 7, 7, 7, 7]]], dtype=torch.int32), tensor([[[8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2]],

#        [[8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2]],

#        [[8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2],
#         [8, 9, 0, 1, 2]]], dtype=torch.int32))


# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor

x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
y = Tensor(np.array([5, 6, 7]).astype(np.int32))
z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
inputs = (x, y, z)
output = mindspore.ops.meshgrid(inputs, indexing='xy')
print(output)
#(Tensor(shape=[3, 4, 5], dtype=Int32, value=
#    [[[1, 1, 1, 1, 1],
#    [2, 2, 2, 2, 2],
#    [3, 3, 3, 3, 3],
#    [4, 4, 4, 4, 4]],
#    [[1, 1, 1, 1, 1],
#    [2, 2, 2, 2, 2],
#    [3, 3, 3, 3, 3],
#    [4, 4, 4, 4, 4]],
#    [[1, 1, 1, 1, 1],
#    [2, 2, 2, 2, 2],
#    [3, 3, 3, 3, 3],
#    [4, 4, 4, 4, 4]]]),
#    Tensor(shape=[3, 4, 5], dtype=Int32, value=
#    [[[5, 5, 5, 5, 5],
#    [5, 5, 5, 5, 5],
#    [5, 5, 5, 5, 5],
#    [5, 5, 5, 5, 5]],
#    [[6, 6, 6, 6, 6],
#    [6, 6, 6, 6, 6],
#    [6, 6, 6, 6, 6],
#    [6, 6, 6, 6, 6]],
#    [[7, 7, 7, 7, 7],
#    [7, 7, 7, 7, 7],
#    [7, 7, 7, 7, 7],
#    [7, 7, 7, 7, 7]]]),
#    Tensor(shape=[3, 4, 5], dtype=Int32, value=
#    [[[8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2]],
#    [[8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2]],
#    [[8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2],
#    [8, 9, 0, 1, 2]]]))
```
