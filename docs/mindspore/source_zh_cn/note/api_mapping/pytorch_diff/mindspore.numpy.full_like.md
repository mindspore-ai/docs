# 比较与torch.full_like的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/mindspore.numpy.full_like.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.full_like

```text
torch.full_like(
    input,
    fill_value,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    memory_format=torch.preserve_format
) -> Tensor
```

更多内容详见[torch.full_like](https://pytorch.org/docs/1.8.1/generated/torch.full_like.html)。

## mindspore.numpy.full_like

```text
mindspore.numpy.full_like(a, fill_value, dtype=None, shape=None) -> Tensor
```

更多内容详见[mindspore.numpy.full_like](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/numpy/mindspore.numpy.full_like.html)。

## 差异对比

PyTorch：返回与填充fill_value的与输入大小和类型相同的张量。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，但支持的输入类型和参数名称有差异。MindSpore算子的输入名称为a，并支持Tensor，list，tuple三种类型的输入。此外，MindSpore该算子比PyTorch新增参数shape，实现重写结果的shape。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 | input         | a         | 功能一致，参数名不同 |
|参数 | 参数1 | - | shape |实现重写shape， PyTorch无此参数 |
|  | 参数2  | fill_value    | fill_value | -                        |
|      | 参数3  | dtype         | dtype      | 功能一致，默认值不同                  |
|      | 参数4  | layout        | -          | 不涉及 |
|      | 参数5  | device        | -          | 不涉及 |
|      | 参数6  | requires_grad | -          | MindSpore无此参数，默认支持反向求导 |
|      | 参数7  | memory_format | -          | 不涉及 |

### 代码示例1

> 对于参数dtype，PyTorch默认值为None，MindSpore默认值为mstype.float32

```python
# PyTorch
import torch

input_tensor_torch = torch.ones((2, 3))
full_value_torch = 1
torch_output = torch.full_like(input_tensor_torch, full_value_torch)
print(torch_output.numpy())
# [[1. 1. 1.]
#  [1. 1. 1.]]

# MindSpore
import mindspore

input_tensor_ms = mindspore.numpy.ones((2, 3))
full_value_ms = [[1., 1., 1.],[1., 1., 1.]]
ms_tensor_output = mindspore.numpy.full_like(input_tensor_ms, full_value_ms)
print(ms_tensor_output)
# [[1. 1. 1.]
#  [1. 1. 1.]]
```
