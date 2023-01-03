# 比较与torch.eye的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/eye.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.eye

```text
torch.eye(
    n,
    m=None,
    *,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False) -> Tensor
```

更多内容详见[torch.eye](https://pytorch.org/docs/1.8.1/generated/torch.eye.html)。

## mindspore.ops.eye

```text
mindspore.ops.eye(n, m, t) -> Tensor
```

更多内容详见[mindspore.ops.eye](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.eye.html)。

## 差异对比

PyTorch： PyTorch中可以在参数中指定一个接受输出的张量、返回的张量的layout、requires_grad以及指定设备。

MindSpore：

PyTorch中参数`m`是可选的，如果没有该参数，那么返回一个列数和行数相同的张量；MindSpore中是必须的。

PyTorch中`dtype`是可选的，如果没有该参数，默认为`torch.float32`；MindSpore中是必须的。

功能上无差异。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                                         |
| ---- | ----- | ------- | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 | n       | n         | -                                                            |
|      | 参数2 | m       | m         | 指定张量的列数。PyTorch中是可选的，如果没有该参数，那么返回一个列数和行数相同的张量；MindSpore中是必须的 |
|       | 参数3 | out | - | 不涉及 |
|      | 参数4 | dtype   | t         | 功能一致，参数名不同，PyTorch中是可选的，如果没有默认为`torch.float32`；MindSpore中是必须的 |
|       | 参数5 | layout | - | 不涉及 |
|       | 参数6 | device | - | 不涉及 |
|       | 参数7 | requires_grad | - | MindSpore无此参数，默认支持反向求导 |

## 差异分析与示例

### 代码示例1

> PyTorch，可以缺省`m`参数。其他功能一致。

```python
# PyTorch
import torch

# 参数m可以缺省，dtype可以缺省
e1 = torch.eye(3)
print(e1.numpy())
# [[1., 0., 0.]
#  [0., 1., 0.]
#  [0., 0., 1.]]

# MindSpore
import mindspore
import mindspore.ops as ops
e1 = ops.eye(3, 3, mindspore.float32)
print(e1)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### 代码示例2

> PyTorch可以缺省`dtype`参数。其他功能一致。

```python
# PyTorch
import torch

# 参数dtype可以缺省
e2 = torch.eye(3, 2)
print(e2.numpy())
# [[1, 0]
#  [0, 1]
#  [0, 0]]

# MindSpore
import mindspore
import mindspore.ops as ops
e2 = ops.eye(3, 2, mindspore.float32)
print(e2)
# [[1. 0.]
#  [0. 1.]
#  [0. 0.]]
```
