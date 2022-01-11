# 比较与torch.Tensor.item的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/item.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## torch.Tensor.item

```python
torch.Tensor.item()
```

更多内容详见[torch.Tensor.item](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.item)。

## mindspore.Tensor.item

```python
mindspore.Tensor.item(index=None)
```

更多内容详见[mindspore.Tensor.item](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/mindspore/mindspore.Tensor.html#mindspore.Tensor.item)。

## 使用方式

PyTorch：返回Tensor的值，适用于只有一个元素的Tensor。

MindSpore：返回Tensor中指定index的值，适用于一个或多个元素的Tensor。

## 代码示例

```python
from mindspore import Tensor
import numpy as np
import torch

x = Tensor(np.array([[1,2,3],[4,5,6]], dtype=np.float32))
print(x.item((0,1)))
# Out：
# 2.0

y = Tensor([1.0])
print(y.item())
# Out:
# [1.]

z = torch.tensor([1.0])
print(z.item())
# Out:
# 1.0
```
