# 比较与torch.normal的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/normal.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.normal

```python
torch.normal(mean, std, *, generator=None, out=None)
torch.normal(mean=0.0, std, *, out=None)
torch.normal(mean, std=1.0, *, out=None)
torch.normal(mean, std, size, *, out=None)
```

更多内容详见[torch.normal](https://pytorch.org/docs/1.8.1/generated/torch.normal.html)。

## mindspore.ops.normal

```python
mindspore.ops.normal(shape, mean, stddev, seed=None)
```

更多内容详见[mindspore.ops.normal](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.normal.html)。

## 差异对比

MindSpore此API功能与PyTorch一致。

PyTorch: 支持四种接口用法。

- `mean` 和 `std` 均为Tensor，要求 `mean` 和 `std` 的成员数量一致，返回值shape和 `mean` 一致。
- `mean` 为float类型，`std` 为Tensor，返回值shape和 `std` 一致。
- `std` 为float类型，`mean` 为Tensor，返回值shape和 `mean` 一致。
- `mean` 和 `std` 均为float类型，返回值shape和 `size` 一致。

MindSpore: `mean` 和 `std` 支持的数据类型是Tensor，返回值的shape由 `shape` , `mean` , `stddev` 三者广播得到。

| 分类       | 子类         | PyTorch      | MindSpore      | 差异          |
| ---------- | ------------ | ------------ | ---------      | ------------- |
| 参数       | 参数 1       | -             | shape         | MindSpore下用于和 `mean` , `stddev` 共同广播得到返回值的shape |
|            | 参数 2       | mean          | mean          | MindSpore下支持的数据类型是Tensor。PyTorch下支持Tensor、float，对应不同用法 |
|            | 参数 3       | std           | stddev        | MindSpore下支持的数据类型是Tensor。PyTorch下支持Tensor、float，对应不同用法 |
|            | 参数 4       | generator     | seed          | MindSpore使用随机数种子生成随机数 |
|            | 参数 5       | size          | -             | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |
|            | 参数 6       | out           | -             | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

## 代码示例 1

> PyTorch下 `mean` 和 `std` 均为Tensor的场景。

```python
# PyTorch
import torch
import numpy as np

mean = torch.tensor(np.array([[3, 4], [5, 6]]), dtype=torch.float32)
stddev = torch.tensor(np.array([[0.2, 0.3], [0.4, 0.5]]), dtype=torch.float32)
output = torch.normal(mean, stddev)
print(output.shape)
# torch.Size([2, 2])

# MindSpore
import mindspore as ms
import numpy as np

shape = (2, 2)
mean = ms.Tensor(np.array([[3, 4], [5, 6]]), ms.float32)
stddev = ms.Tensor(np.array([[0.2, 0.3], [0.4, 0.5]]), ms.float32)
output = ms.ops.normal(shape, mean, stddev)
print(output.shape)
# (2, 2)
```

## 代码示例 2

> PyTorch下 `mean` 为float， `std` 为Tensor的场景。

```python
# PyTorch
import torch
import numpy as np

mean = 3.0
stddev = torch.tensor(np.array([[0.2, 0.3], [0.4, 0.5]]), dtype=torch.float32)
output = torch.normal(mean, stddev)
print(output.shape)
# torch.Size([2, 2])

# MindSpore
import mindspore as ms
import numpy as np

shape = (2, 2)
mean = ms.Tensor(3.0, ms.float32)
stddev = ms.Tensor(np.array([[0.2, 0.3], [0.4, 0.5]]), ms.float32)
output = ms.ops.normal(shape, mean, stddev)
print(output.shape)
# (2, 2)
```

## 代码示例 3

> PyTorch下 `mean` 为Tensor， `std` 为float的场景。

```python
# PyTorch
import torch
import numpy as np

mean = torch.tensor(np.array([[3, 4], [5, 6]]), dtype=torch.float32)
stddev = 1.0
output = torch.normal(mean, stddev)
print(output.shape)
# torch.Size([2, 2])

# MindSpore
import mindspore as ms
import numpy as np

shape = (2, 2)
mean = ms.Tensor(np.array([[3, 4], [5, 6]]), ms.float32)
stddev = ms.Tensor(1.0, ms.float32)
output = ms.ops.normal(shape, mean, stddev)
print(output.shape)
# (2, 2)
```

## 代码示例 4

> PyTorch下 `mean` 和 `std` 均为float的场景。

```python
# PyTorch
import torch
import numpy as np

mean = 3.0
stddev = 1.0
size = (2, 2)
output = torch.normal(mean, stddev, size)
print(output.shape)
# torch.Size([2, 2])

# MindSpore
import mindspore as ms
import numpy as np

shape = (2, 2)
mean = ms.Tensor(3.0, ms.float32)
stddev = ms.Tensor(1.0, ms.float32)
output = ms.ops.normal(shape, mean, stddev)
print(output.shape)
# (2, 2)
```
