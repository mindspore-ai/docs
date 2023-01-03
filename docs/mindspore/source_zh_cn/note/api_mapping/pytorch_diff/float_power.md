# 比较与torch.float_power的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/float_power.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.float_power

```python
torch.float_power(input, exponent, *, out=None) → Tensor
```

更多内容详见[torch.float_power](https://pytorch.org/docs/1.8.1/generated/torch.float_power.html)。

## mindspore.ops.float_power

```python
mindspore.ops.float_power(x, exponent)
```

更多内容详见[mindspore.ops.float_power](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.float_power.html#mindspore.ops.float_power)。

## 差异对比

PyTorch：将输入tensor提高到双精度计算指数幂。如果两个输入都不是复数，则返回torch.float64张量，如果一个或多个输入是复数，则返回torch.complex128张量。

MindSpore：

- 如果两个输入都不是复数，MindSpore此API实现功能与PyTorch一致，仅参数名不同；
- 如果输入中有复数类型，MindSpore不会进行精度提升，目前复数运算只支持CPU；
    - 当输入是两个复数Tensor时，MindSpore要求两个Tensor类型相同，返回值与输入的类型相同；
    - 当输入是一个复数Tensor和一个scalar时，MindSpore的返回值与输入Tensor的类型相同；
    - 当输入是一个复数Tensor和一个实数Tensor时，MindSpore目前不支持此种运算。

| 分类 | 子类  | PyTorch | MindSpore | 差异                 |
| ---- | ----- | ------- | --------- | -------------------- |
| 参数 | 参数1 | input   | x         | 功能一致，参数名不同 |
|      | 参数2 | exponent | exponent | 功能一致 |
|      | 参数3 | out     | -         | 不涉及              |

## 代码示例1

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

## 代码示例2

> 当输入是复数类型，MindSpore仅支持CPU平台上输入为两个复数类型的Tensor或一个复数Tensor一个scalar，且返回值类型与输入的复数Tensor类型相同。

```python
import numpy as np
input_np = np.array([(2., 3.), (3., 4.), (4., 5.)], np.complex64)
# PyTorch
import torch
input = torch.from_numpy(input_np)
out_torch = torch.float_power(input, 2.)
print(out_torch.detach().numpy(), out_torch.detach().numpy().dtype)
# [[ 4.+0.j  9.+0.j]
#  [ 9.+0.j 16.+0.j]
#  [16.+0.j 25.+0.j]] complex128

# MindSpore
import mindspore
from mindspore import Tensor, ops
x = Tensor(input_np)
output = ops.float_power(x, 2.)
print(output.asnumpy())
# [[ 4.      +0.j  9.      +0.j]
#  [ 9.      +0.j 16.      +0.j]
#  [16.      +0.j 25.000002+0.j]] complex64
```
