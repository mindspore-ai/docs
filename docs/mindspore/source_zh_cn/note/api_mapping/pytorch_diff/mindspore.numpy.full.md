# 比较与torch.full的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/mindspore.numpy.full.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.full

```text
torch.full(
    size,
    fill_value,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False
) -> Tensor
```

更多内容详见[torch.full](https://pytorch.org/docs/1.8.1/generated/torch.full.html)。

## mindspore.numpy.full

```text
mindspore.numpy.full(shape, fill_value, dtype=None) -> Tensor
```

更多内容详见[mindspore.numpy.full](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/numpy/mindspore.numpy.full.html)。

## 差异对比

PyTorch：返回用fill_value填充的给定大小的张量。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，但参数名不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | size | shape |功能一致，参数名不同 |
| | 参数2 | fill_value | fill_value | 功能一致 |
|  | 参数3 | dtype         | dtype     | 功能一致       |
| | 参数4 | out           | -         | 不涉及 |
| | 参数5 | layout | - | 不涉及 |
| | 参数6 | device | - | 不涉及 |
| | 参数7 | requires_grad | - | MindSpore无此参数，默认支持反向求导 |

### 代码示例1

> 对于参数fill_value，PyTorch的full算子支持类型为number，MindSpore支持类型包括int，float，bool，list，tuple。当MindSpore的full算子输入类型为list或tuple时，注意其shape要符合广播规则。

```python
# PyTorch
import torch

torch_output = torch.full((2, 3), 1)
print(torch_output.numpy())
# [[1 1 1]
#  [1 1 1]]

# MindSpore
import mindspore

full_value = [[1, 1, 1],[1, 1, 1]]
ms_tensor_output = mindspore.numpy.full((2, 3), full_value)
print(ms_tensor_output)
# [[1 1 1]
#  [1 1 1]]
```
