# 比较与torch.Tensor.max的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/tensor_max.md)

## torch.Tensor.max

```python
torch.Tensor.max(dim=None, keepdim=False)
```

更多内容详见[torch.Tensor.max](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.max)。

## mindspore.Tensor.max

```python
mindspore.Tensor.max(axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
```

更多内容详见[mindspore.Tensor.max](https://www.mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore/Tensor/mindspore.Tensor.max.html)。

## 差异对比

MindSpore在PyTorch的基础上，兼容了Numpy的入参 `initial` 和 `where`，新增了参数return_indices用于控制是否返回索引。

| 分类  | 子类  | PyTorch | MindSpore | 差异         |
|-----|-----|---------|-----------|------------|
| 输入  | 输入1 | dim     | axis      | 功能一致，参数名不同 |
|     | 输入2 | keepdim | keepdims  | 功能一致，参数名不同 |
|     | 输入3 | -      |initial    | 不涉及        |
|     | 输入4 |  -     |where    | 不涉及        |
|     | 输入5 |  -     |return_indices    | 不涉及        |

### 代码示例1

不指定维度时，两API实现功能一致。

```python
import mindspore as ms
import torch
import numpy as np

np_x = np.array([[-0.0081, -0.3283, -0.7814, -0.0934],
                 [1.4201, -0.3566, -0.3848, -0.1608],
                 [-0.0446, -0.1843, -1.1348, 0.5722],
                 [-0.6668, -0.2368, 0.2790, 0.0453]]).astype(np.float32)
# mindspore
input_x = ms.Tensor(np_x)
output = input_x.max()
print(output)
# 1.4201

# torch
input_x = torch.tensor(np_x)
output = input_x.max()
print(output)
# tensor(1.4201)
```

### 代码示例2

指定维度时，MindSpore默认不返回索引，需手动指定。

```python
import mindspore as ms
import torch
import numpy as np

np_x = np.array([[-0.0081, -0.3283, -0.7814, -0.0934],
                 [1.4201, -0.3566, -0.3848, -0.1608],
                 [-0.0446, -0.1843, -1.1348, 0.5722],
                 [-0.6668, -0.2368, 0.2790, 0.0453]]).astype(np.float32)
# mindspore
input_x = ms.Tensor(np_x)
values, indices = input_x.max(axis=1, return_indices=True)
print(values)
# [-0.0081  1.4201  0.5722  0.279 ]
print(indices)
# [0 0 3 2]

# torch
input_x = torch.tensor(np_x)
values, indices = input_x.max(dim=1)
print(values)
# tensor([-0.0081,  1.4201,  0.5722,  0.2790])
print(indices)
# tensor([0, 0, 3, 2])
```