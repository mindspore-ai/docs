# 比较与torch.Tensor.item的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/item.md)

## torch.Tensor.item

```python
torch.Tensor.item()
```

更多内容详见[torch.Tensor.item](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.item)。

## mindspore.Tensor.item

```python
mindspore.Tensor.item(index=None)
```

更多内容详见[mindspore.Tensor.item](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore/Tensor/mindspore.Tensor.item.html#mindspore.Tensor.item)。

## 使用方式

PyTorch：返回Tensor的值，适用于只有一个元素的Tensor。

MindSpore：返回Tensor中指定index的值，适用于一个或多个元素的Tensor。

## 代码示例

```python
import mindspore as ms
import numpy as np
import torch

x = ms.Tensor(np.array([[1,2,3],[4,5,6]], dtype=np.float32))
print(x.item((0,1)))
# Out：
# 2.0

y = ms.Tensor([1.0])
print(y.item())
# Out:
# 1.0

z = torch.tensor([1.0])
print(z.item())
# Out:
# 1.0
```
