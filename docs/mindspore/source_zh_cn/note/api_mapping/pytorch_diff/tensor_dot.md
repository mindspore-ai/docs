# 比较与torch.dot的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/tensor_dot.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.dot

```python
torch.dot(input, other, *, out=None)
```

更多内容详见[torch.dot](https://pytorch.org/docs/1.8.1/generated/torch.dot.html)。

## mindspore.ops.tensor_dot

```python
mindspore.ops.tensor_dot(x1, x2, axes)
```

更多内容详见[mindspore.ops.tensor_dot](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.tensor_dot.html#mindspore.ops.tensor_dot)。

## 使用方式

PyTorch：计算两个相同shape的tensor的点乘（内积），仅支持1D。支持的输入数据类型包括uint8、int8/16/32/64、float32/64。

MindSpore：计算两个tensor在任意轴上的点乘，支持任意维度的tensor，但指定的轴对应的形状要相等。当输入为1D， 轴设定为1时，和PyTorch的功能一致。支持的输入数据类型为float16或float32。

| 分类       | 子类         | PyTorch      | MindSpore  | 差异          |
| ---------- | ------------ | ------------ | ---------  | ------------- |
| 参数       | 参数 1       | input         | x1        | 功能一致，参数名不同 |
|            | 参数 2       | other         | x2        | 功能一致，参数名不同 |
|            | 参数 3       | out           | -         | 不涉及        |
|            | 参数 4       | -             | axes      | 当输入为1D，axes设定为1时，和PyTorch的功能一致 |

## 代码示例

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, tensor of any dimension will be supported.
# And parameters will be set to specify how to compute among dimensions.
input_x1 = ms.Tensor(np.array([2, 3, 4]), ms.float32)
input_x2 = ms.Tensor(np.array([2, 1, 3]), ms.float32)
output = ops.tensor_dot(input_x1, input_x2, 1)
print(output)
# Out：
# 19.0

# In torch, only 1D tensor's computation will be supported.
input_x1 = torch.tensor([2, 3, 4])
input_x2 = torch.tensor([2, 1, 3])
output = torch.dot(input_x1, input_x2)
print(output)
# Out：
# tensor(19)
```
