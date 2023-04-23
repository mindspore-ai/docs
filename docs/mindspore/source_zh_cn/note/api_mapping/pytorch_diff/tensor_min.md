# 比较与torch.Tensor.min的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/tensor_min.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.Tensor.min

```python
torch.Tensor.min(dim=None, keepdim=False)
```

更多内容详见[torch.Tensor.min](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.min)。

## mindspore.Tensor.min

```python
mindspore.Tensor.min(axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
```

更多内容详见[mindspore.Tensor.min](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/Tensor/mindspore.Tensor.min.html)。

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
output = input_x.min()
print(output)
# -1.1348

# torch
input_x = torch.tensor(np_x)
output = input_x.min()
print(output)
# tensor(-1.1348)
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
values, indices = input_x.min(axis=1, return_indices=True)
print(values)
# [-0.7814 -0.3848 -1.1348 -0.6668]
print(indices)
# [2 2 2 0]

# torch
input_x = torch.tensor(np_x)
values, indices = input_x.min(dim=1)
print(values)
# tensor([-0.7814, -0.3848, -1.1348, -0.6668])
print(indices)
# tensor([2, 2, 2, 0])
```