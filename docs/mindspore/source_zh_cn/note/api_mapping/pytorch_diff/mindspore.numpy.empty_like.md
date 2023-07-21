# 比较与torch.empty_like的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/mindspore.numpy.empty_like.md)

## torch.empty_like

```text
torch.empty_like(
    input,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    memory_format=torch.preserve_format
) -> Tensor
```

更多内容详见[torch.empty_like](https://pytorch.org/docs/1.8.1/generated/torch.empty_like.html)。

## mindspore.numpy.empty_like

```text
mindspore.numpy.empty_like(prototype, dtype=None, shape=None) -> Tensor
```

更多内容详见[mindspore.numpy.empty_like](https://mindspore.cn/docs/zh-CN/master/api_python/numpy/mindspore.numpy.empty_like.html)。

## 差异对比

PyTorch：返回与输入大小和类型相同的未初始化张量，input只支持Tensor类型输入。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，但支持的输入类型和参数名称有差异。MindSpore算子的输入名称为prototype，并支持Tensor，list，tuple三种类型的输入。此外，MindSpore该算子比PyTorch新增参数shape，实现重写结果的shape。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 |input | prototype | 功能一致，MindSpore支持更多输入类型 |
|参数 | 参数1 | dtype         | dtype     | - |
|  | 参数2 | -             | shape     | 实现重写shape，PyTorch无此参数 |
| | 参数3 | layout | - | 不涉及 |
| | 参数4 | device | - | 不涉及 |
| | 参数5 | requires_grad | - | MindSpore无此参数，默认支持反向求导 |
| | 参数6 | memory_format | - | 不涉及 |

### 代码示例1

> 对于参数shape，PyTorch的empty_like算子无此参数，MindSpore的shape参数默认值为None，通过该参数可以实现对结果shape的重写。

```python
# PyTorch
import torch

input_torch = torch.ones((2, 3))
torch_output = torch.empty_like(input_torch)
print(list(torch_output.shape))
# [2, 3]

# MindSpore
import mindspore

input_ms = mindspore.numpy.ones((4,1,2))
ms_output = mindspore.numpy.empty_like(input_ms, shape=[2, 3])
print(ms_output.shape)
# (2, 3)
```

### 代码示例2

> PyTorch的empty_like算子支持输入类型为Tensor，但MindSpore支持三种输入类型Tensor，list，tuple。在输入为数组类型时，数组在维度上必须具有相同的大小。如果输入类型不是Tensor，则默认数据类型为float32（如果未提供dtype）。

```python
# PyTorch
import torch

input_tensor_torch = torch.ones((2, 3))
torch_output = torch.empty_like(input_tensor_torch)
print(list(torch_output.shape))
# [2, 3]

# MindSpore
import mindspore

input_tensor_ms = mindspore.numpy.ones((2, 3))
ms_tensor_output = mindspore.numpy.empty_like(input_tensor_ms)
print(ms_tensor_output.shape)
# (2, 3)

input_list_ms = [[1, 2, 3],[4, 5, 6]]
ms_list_output = mindspore.numpy.empty_like(input_list_ms)
print(ms_list_output.shape)
# (2, 3)

input_tuple_ms = ((1, 2, 3),(4, 5, 6))
ms_tuple_output = mindspore.numpy.empty_like(input_tuple_ms)
print(ms_tuple_output.shape)
# (2, 3)
```
