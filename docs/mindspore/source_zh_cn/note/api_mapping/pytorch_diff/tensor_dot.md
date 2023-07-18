# 比较与torch.dot的差异

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/tensor_dot.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png"></a>

## torch.dot

```python
torch.dot(input, other, *, out=None)
```

更多内容详见[torch.dot](https://pytorch.org/docs/1.8.1/generated/torch.dot.html)。

## mindspore.ops.tensor_dot

```python
mindspore.ops.tensor_dot(x1, x2, axes)
```

更多内容详见[mindspore.ops.tensor_dot](https://mindspore.cn/docs/zh-CN/r2.1/api_python/ops/mindspore.ops.tensor_dot.html#mindspore.ops.tensor_dot)。

## 使用方式

MindSpore此API功能与PyTorch不一致。

PyTorch：计算两个相同shape的tensor的点乘（内积），仅支持1D。支持的输入数据类型包括uint8、int8/16/32/64、float32/64。

MindSpore：计算两个tensor在任意轴上的点乘，支持任意维度的tensor，但指定的轴对应的形状要相等。当输入为1D， 轴设定为1时，和PyTorch的功能一致。支持的输入数据类型为float16或float32。

| 分类       | 子类         | PyTorch      | MindSpore  | 差异          |
| ---------- | ------------ | ------------ | ---------  | ------------- |
| 参数       | 参数 1       | input         | x1        | 参数名不同     |
|            | 参数 2       | other         | x2        | 参数名不同     |
|            | 参数 3       | out           | -         | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.1/note/api_mapping/pytorch_api_mapping.html#通用差异参数表)  |
|            | 参数 4       | -             | axes      | 当输入为1D，axes设定为1时，和PyTorch的功能一致 |

## 代码示例 1

> 输入的数据类型是int，输出的数据类型也是int。

```python
import torch
input_x1 = torch.tensor([2, 3, 4], dtype=torch.int32)
input_x2 = torch.tensor([2, 1, 3], dtype=torch.int32)
output = torch.dot(input_x1, input_x2)
print(output)
# tensor(19, dtype=torch.int32)
print(output.dtype)
# torch.int32

# MindSpore目前无法支持该功能。
```

## 代码示例 2

> 输入的数据类型是float，输出的数据类型也是float。

```python
import torch
input_x1 = torch.tensor([2, 3, 4], dtype=torch.float32)
input_x2 = torch.tensor([2, 1, 3], dtype=torch.float32)
output = torch.dot(input_x1, input_x2)
print(output)
# tensor(19.)
print(output.dtype)
# torch.float32

import mindspore as ms
import mindspore.ops as ops
import numpy as np
input_x1 = ms.Tensor(np.array([2, 3, 4]), ms.float32)
input_x2 = ms.Tensor(np.array([2, 1, 3]), ms.float32)
output = ops.tensor_dot(input_x1, input_x2, 1)
print(output)
# 19.0
print(output.dtype)
# Float32
```
