# 比较与torch.full的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/full.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

## mindspore.ops.full

```text
mindspore.ops.full(size, fill_value, *, dtype=None) -> Tensor
```

更多内容详见[mindspore.ops.full](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.full.html)。

## 差异对比

PyTorch：返回用fill_value填充的给定大小的张量。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，但参数名不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | size | size |功能一致 |
| | 参数2 | fill_value | fill_value | 功能一致 |
|  | 参数3 | dtype         | dtype     | 功能一致       |
| | 参数4 | out           | -         | 不涉及 |
| | 参数5 | layout | - | 不涉及 |
| | 参数6 | device | - | 不涉及 |
| | 参数7 | requires_grad | - | MindSpore无此参数，默认支持反向求导 |

### 代码示例1

> 对于参数fill_value，PyTorch的full算子支持类型为number，MindSpore不支持复数类型。

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
ms_tensor_output = mindspore.ops.full((2, 3), full_value)
print(ms_tensor_output)
# [[1 1 1]
#  [1 1 1]]
```
