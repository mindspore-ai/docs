# 比较与torch.Tensor.stride的功能差异

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/stride.md)

## torch.Tensor.stride

```python
torch.Tensor.stride(dim)
```

## mindspore.Tensor.strides

```python
mindspore.Tensor.strides()
```

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
