# 比较与torch.cross的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/cross.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.cross

```python
class torch.cross(
    input,
    other,
    dim=-1,
    out=None
)
```

更多内容详见 [torch.cross](https://pytorch.org/docs/1.5.0/torch.html#torch.cross)。

## mindspore.numpy.cross

```python
class mindspore.numpy.cross(
    a,
    b,
    axisa=- 1,
    axisb=- 1,
    axisc=- 1,
    axis=None
)
```

更多内容详见 [mindspore.numpy.cross](https://mindspore.cn/docs/api/zh-CN/master/api_python/numpy/mindspore.numpy.cross.html#mindspore.numpy.cross)。

## 使用方式

PyTorch: 返回input和other维度dim中向量的叉积。输入必须具有相同的大小，并且它们的dim维度的大小应为3。如果未给出dim，则默认为找到的第一个大小为3的维度。

MindSpore: 如果a和b是向量数组，则默认情况下，向量由a和b的最后一个轴定义，这些轴的维度可以是2或3。当a或b的维数为2时，假设输入向量的第三个分量为零，并相应地计算叉积。如果两个输入向量的维度均为2，则返回叉积的z分量。

## 代码示例

```python
import mindspore.numpy as np
import torch

# MindSpore
x = np.array([[1,2,3], [4,5,6]])
y = np.array([[4,5,6], [1,2,3]])
output = np.cross(x, y)
print(output)
# [[-3  6 -3]
# [ 3 -6  3]]
output = np.cross(x, y, axisc=0)
print(output)
# [[-3  3]
# [ 6 -6]
# [-3  3]]
x = np.array([[1,2], [4,5]])
y = np.array([[4,5], [1,2]])
print(np.cross(x, y))
# Tensor(shape=[2], dtype=Int32, value= [-3,  3])

# PyTorch
a = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.int8)
b = torch.tensor([[4,5,6], [1,2,3]], dtype=torch.int8)
print(torch.cross(a, b, dim=1))
# tensor([[-3,  6, -3],
#         [ 3, -6,  3]], dtype=torch.int8)
print(torch.cross(a, b))
# tensor([[-3,  6, -3],
#         [ 3, -6,  3]], dtype=torch.int8)
a = torch.tensor([[1,2], [4,5]], dtype=torch.int8)
b = torch.tensor([[4,5], [1,2]], dtype=torch.int8)
print(torch.cross(a, b))
# RuntimeError: no dimension of size 3 in input
```
