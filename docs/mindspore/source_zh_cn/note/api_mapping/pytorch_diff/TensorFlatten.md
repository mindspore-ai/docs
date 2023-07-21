# 比较与torch.Tensor.flatten的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.10/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TensorFlatten.md)

## torch.Tensor.flatten

```python
torch.Tensor.flatten(input, start_dim=0, end_dim=-1)
```

更多内容详见[torch.Tensor.flatten](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.flatten)。

## mindspore.Tensor.flatten

```python
mindspore.Tensor.flatten(order="C")
```

更多内容详见[mindspore.Tensor.flatten](https://www.mindspore.cn/docs/en/r1.10/api_python/mindspore/Tensor/mindspore.Tensor.flatten.html#mindspore.Tensor.flatten)。

## 使用方式

`torch.flatten`通过入参`start_dim`，`end_dim`限制需要扩展的维度范围。

`mindspore.Tensor.flatten`通过`order`为"C"或"F"确定优先按行还是列展平。

## 代码示例

```python
import mindspore as ms

a = ms.Tensor([[1,2], [3,4]], ms.float32)
print(a.flatten())
# [1. 2. 3. 4.]
print(a.flatten('F'))
# [1. 3. 2. 4.]

import torch

b = torch.tensor([[[1, 2],[3, 4]],[[5, 6],[7, 8]]])
print(torch.Tensor.flatten(b))
# tensor([1, 2, 3, 4, 5, 6, 7, 8])
print(torch.Tensor.flatten(b, start_dim=1))
# tensor([[1, 2, 3, 4],
#         [5, 6, 7, 8]])
```
