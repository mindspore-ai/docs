# 比较与torch.float_power的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/float_power.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

> `mindspore.Tensor.float_power` 和 `torch.Tensor.float_power` 的功能差异，参考 `mindspore.ops.float_power` 和 `torch.float_power` 的功能差异比较。

## torch.float_power

```python
torch.float_power(input, exponent, *, out=None) -> Tensor
```

更多内容详见[torch.float_power](https://pytorch.org/docs/1.8.1/generated/torch.float_power.html)。

## mindspore.ops.float_power

```python
mindspore.ops.float_power(x, exponent)
```

更多内容详见[mindspore.ops.float_power](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.float_power.html#mindspore.ops.float_power)。

## 差异对比

PyTorch：将输入tensor提高到双精度计算指数幂。如果两个输入都不是复数，则返回torch.float64张量，如果一个或多个输入是复数，则返回torch.complex128张量。

MindSpore：如果两个输入都是实数，MindSpore此API实现功能与PyTorch一致，仅参数名不同。目前不支持复数运算。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同 |
|      | 参数2 | exponent | exponent | 功能一致 |
|      | 参数3 | out     | -         | 不涉及              |

## 代码示例

> 当输入是实数类型，两API功能一致，用法相同。

```python
import numpy as np
input_np = np.array([2., 3., 4.], np.float32)
# PyTorch
import torch
input = torch.from_numpy(input_np)
out_torch = torch.float_power(input, 2.)
print(out_torch.detach().numpy())
# [ 4.  9. 16.]

# MindSpore
import mindspore
from mindspore import Tensor, ops
x = Tensor(input_np)
output = ops.float_power(x, 2.)
print(output.asnumpy())
# [ 4.  9. 16.]
```
