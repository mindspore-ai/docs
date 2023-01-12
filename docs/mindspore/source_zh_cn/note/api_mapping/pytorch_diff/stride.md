# 比较与torch.Tensor.stride的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/stride.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png"></a>

## torch.Tensor.stride

```python
torch.Tensor.stride(dim)
```

更多内容详见[torch.Tensor.stride](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.stride)。

## mindspore.Tensor.strides

```python
mindspore.Tensor.strides()
```

更多内容详见[mindspore.Tensor.strides](https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/Tensor/mindspore.Tensor.strides.html#mindspore.Tensor.strides)。

## 使用方式

PyTorch：每一维中遍历所需要经过的元素个数，返回类型为元组。

MindSpore：每一维中遍历所需要经过的字节数，返回类型为元组。

## 代码示例

```python
import mindspore as ms

a = ms.Tensor([[1, 2, 3], [7, 8, 9]])
print(a.strides)
# out:
# (24, 8)

import torch as tc

b = tc.tensor([[1, 2, 3], [7, 8, 9]])
print(b.stride())
# out:
# (3, 1)
```
