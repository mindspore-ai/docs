# 比较与torch.Tensor.masked_scatter的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/masked_scatter.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.Tensor.masked_scatter

```python
torch.Tensor.masked_scatter(mask, tensor) -> Tensor
```

更多内容详见[torch.Tensor.masked_scatter](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.masked_scatter)。

## mindspore.Tensor.masked_scatter

```python
mindspore.Tensor.masked_scatter(mask, tensor) -> Tensor
```

更多内容详见[mindspore.Tensor.masked_scatter](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.masked_scatter.html)。

## 差异对比

PyTorch：返回一个Tensor。根据 `mask` ，使用 `tensor` 中的值，更新Tensor本身的值。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。但是PyTorch支持 `mask` 与Tensor本身的双向广播，
MindSpore只支持 `mask` 广播到Tensor本身。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                                         |
| ---- | ----- | ------- | --------- | -----------------------------------------------------------|
| 参数 | 参数1 | mask | mask | PyTorch支持 `mask` 与Tensor本身的双向广播，MindSpore只支持 `mask` 广播到Tensor本身 |
|      | 参数2 | tensor | tensor | - |

### 代码示例1

```python
# PyTorch
import torch

self = torch.tensor([0, 0, 0, 0, 0])
mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]])
source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
output = self.masked_scatter(mask, source)
print(output)
# tensor([[0, 0, 0, 0, 1],
#         [2, 3, 0, 4, 5]])

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

self = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), mindspore.int32)
mask = Tensor(np.array([[False, False, False, True, True], [True, True, False, True, True]]), mindspore.bool_)
source = Tensor(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), mindspore.int32)
output = self.masked_scatter(mask, source)
print(output)
# [[0 0 0 0 1],
#  [2 3 0 4 5]]
```

### 代码示例2

```python
# PyTorch
import torch

self = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
mask = torch.tensor([0, 0, 0, 1, 1])
source = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
output = self.masked_scatter(mask, source)
print(output)
# tensor([[0, 0, 0, 0, 1],
#         [0, 0, 0, 2, 3]])

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

self = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), mindspore.int32)
mask = Tensor(np.array([False, False, False, True, True]), mindspore.bool_)
source = Tensor(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), mindspore.int32)
output = self.masked_scatter(mask, source)
print(output)
# [[0 0 0 0 1],
#  [0 0 0 2 3]]
```
