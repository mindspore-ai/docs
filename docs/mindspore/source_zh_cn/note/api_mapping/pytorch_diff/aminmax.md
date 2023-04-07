# 比较与torch.aminmax的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/aminmax.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.aminmax

```text
torch.aminmax(input, *, dim=None, keepdim=False, out=None) -> Tensor
```

更多内容详见[torch.aminmax](https://pytorch.org/docs/1.12/generated/torch.aminmax.html)。

## mindspore.ops.aminmax

```text
mindspore.ops.aminmax(input, *, axis=0, keepdims=False) -> Tensor
```

更多内容详见[mindspore.ops.aminmax](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.aminmax.html)。

## 差异对比

PyTorch：根据指定 `dim`，求 `input` 的最小值和最大值元素。`dim` 默认值为None，求输入所有值的极值。

MindSpore：实现功能与PyTorch基本一致。不过MindSpore的 `axis` 有默认值：0，在第零维上求取极值。而PyTorch默认情况下计算所有维度的极值。MindSpore可以通过组合 amin 和 amax 两个接口实现同样的功能。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | input   | input        | -                   |
|      | 参数2 | dim   | axis      | MindSpore的 `axis` 有默认值：0，在第零维上求取极值。PyTorch的 `dim` 默认值为None，求输入所有值的极值。 |
|      | 参数3 | keepdim   | keepdims | - |
|      | 参数4 | out   | -         | 不涉及 |

### 代码示例1

> 在输入所有维度上计算最大值和最小值，PyTorch可以通过使用 `dim` 的默认值来实现这一功能。MindSpore可以通过调用 amin 和 amax 组合实现同样的功能。

```python
# PyTorch
import torch

input = torch.tensor([[3, 1, 4], [1, 5, 9]], dtype=torch.float32)
output = torch.aminmax(input, keepdim=True)
print(output)
# torch.return_types.aminmax(
# min=tensor([[1.]]),
# max=tensor([[9.]]))

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor([[3, 1, 4], [1, 5, 9]], dtype=mindspore.float32)
output_min = ops.amin(x)
output_max = ops.amax(x)
print((output_min, output_max))
# (Tensor(shape=[], dtype=Float32, value= 1), Tensor(shape=[], dtype=Float32, value= 9))
```

### 代码示例2

> 在指定维度上计算最小值和最大值，只需要将PyTorch的 `dim` 和MindSpore的 `axis` 设置同样的值即可。

```python
# PyTorch
import torch

input = torch.tensor([[3, 1, 4], [1, 5, 9]], dtype=torch.int32)
output = torch.aminmax(input, dim=0, keepdim=True)
print(output)
# torch.return_types.aminmax(
# min=tensor([[1, 1, 4]], dtype=torch.int32),
# max=tensor([[3, 5, 9]], dtype=torch.int32))

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor([[3, 1, 4], [1, 5, 9]], dtype=mindspore.int32)
output = ops.aminmax(x, axis=0, keepdims=True)
print(output)
# (Tensor(shape=[1, 3], dtype=Int32, value=
# [[1, 1, 4]]), Tensor(shape=[1, 3], dtype=Int32, value=
# [[3, 5, 9]]))
```