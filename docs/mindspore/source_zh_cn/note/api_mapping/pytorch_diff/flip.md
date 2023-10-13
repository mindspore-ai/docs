# 比较与torch.Tensor.flip的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/flip.md)

## torch.Tensor.flip

```python
torch.Tensor.flip(dims)
```

更多内容详见[torch.Tensor.flip](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.flip)。

## mindspore.Tensor.flip

```python
mindspore.Tensor.flip(dims)
```

更多内容详见[mindspore.Tensor.flip](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/mindspore/Tensor/mindspore.Tensor.flip.html)。

## 使用方式

PyTorch：torch.Tensor.flip接口与torch.flip本身有差异，相比torch.flip，Tensor.flip额外支持了dim入参为int的场景。

MindSpore：mindspore.flip与mindspore.Tensor.flip接口功能与torch.flip一致，均不支持入参为int的场景。

| 分类  | 子类  | PyTorch   | MindSpore | 差异         |
|-----|-----|-----------|-----------|------------|
| 参数 | 参数1 | dims     | dims      | 功能一致，MindSpore不支持int入参   |

## 代码示例

```python
# PyTorch
import numpy as np
import torch
input = torch.tensor(np.arange(1, 9).reshape((2, 2, 2)))
output = input.flip(1)
print(output)
# tensor([[[3, 4],
#          [1, 2]],

#         [[7, 8],
#          [5, 6]]])

# MindSpore
import mindspore as ms
import mindspore.ops as ops

input = ms.Tensor(np.arange(1, 9).reshape((2, 2, 2)))
output = input.flip((1, ))
print(output)
# [[[3 4]
#   [1 2]]

#  [[7 8]
#   [5 6]]]
```
