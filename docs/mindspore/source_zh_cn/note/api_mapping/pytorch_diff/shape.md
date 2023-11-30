# 比较与torch.Tensor.size的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/shape.md)

## torch.Tensor.size

```text
torch.Tensor.size() -> Tensor
```

更多内容详见[torch.Tensor.size](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.size)。

## mindspore.Tensor.shape

```text
mindspore.Tensor.shape
```

更多内容详见[mindspore.Tensor.shape](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.shape.html)。

## 差异对比

PyTorch：size()方法，返回Tensor的shape。

MindSpore：功能一致，但是mindspore.Tensor.shape为属性，不为方法。

### 代码示例

```python
# PyTorch
import torch

input = torch.randn(3, 4, 5)
print(input.size())
# torch.Size([3, 4, 5])

# MindSpore
import mindspore.ops as ops

input = ops.randn(3, 4, 5)
print(input.shape)
# (3, 4, 5)
```
