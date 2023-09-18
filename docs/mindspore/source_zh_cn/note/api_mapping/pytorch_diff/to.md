# 比较与torch.Tensor.to的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/to.md)

## torch.Tensor.to

```python
torch.Tensor.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
torch.Tensor.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
torch.Tensor.to(other, non_blocking=False, copy=False) -> Tensor
```

更多内容详见[torch.Tensor.to](https://pytorch.org/docs/1.8.1/tensors.html?#torch.Tensor.to)。

## mindspore.Tensor.to

```python
mindspore.Tensor.to(dtype)
```

更多内容详见[mindspore.Tensor.to](https://mindspore.cn/docs/zh-CN/r2.1/api_python/mindspore/Tensor/mindspore.Tensor.to.html)。

## 使用方式

MindSpore此API功能与PyTorch不一致。

PyTorch：支持三种接口用法。

- 当仅提供 `dtype` 参数时，该接口返回指定数据类型的Tensor，此时用法和MindSpore一致。
- 当提供了 `device` 参数时，该接口返回的Tensor指定了设备，MindSpore不支持该能力。
- 当提供了 `other` 时，该接口返回和 `other` 相同数据类型和设备的Tensor，MindSpore不支持该能力。

MindSpore：仅支持 `dtype` 参数，返回指定数据类型的Tensor。

| 分类       | 子类         | PyTorch      | MindSpore  | 差异          |
| ---------- | ------------ | ------------ | ---------  | ------------- |
| 参数       | 参数 1       | dtype         | dtype      | 使用对应框架下的数据类型 |
|            | 参数 2       | device        | -         | PyTorch指定设备，MindSpore不支持该功能 |
|            | 参数 3       | other         | -         | PyTorch指定使用的Tensor，MindSpore不支持该功能 |
|            | 参数 4       | non_blocking  | -         | PyTorch用于CPU和GPU之间的异步拷贝，MindSpore不支持该功能 |
|            | 参数 5       | copy          | -         | PyTorch用于强制创建新的Tensor，MindSpore不支持该功能 |
|            | 参数 6       | memory_format | -         | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.1/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

## 代码示例 1

> 仅指定 `dtype` 。

```python
# PyTorch
import torch
import numpy as np
input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
input_x = torch.tensor(input_np)
dtype = torch.int32
output = input_x.to(dtype)
print(output.dtype)
# torch.int32

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np
input_x = Tensor(input_np)
dtype = mindspore.int32
output = input_x.to(dtype)
print(output.dtype)
# Int32
```

## 代码示例 2

> 指定 `device` 。

```python
# PyTorch
import torch
import numpy as np
input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
input_x = torch.tensor(input_np)
device = torch.device('cpu')
output = input_x.to(device)
print(output.device)
# cpu

# MindSpore目前无法支持该功能。
```

## 代码示例 3

> 指定另一个Tensor。

```python
# PyTorch
import torch
import numpy as np
input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
input_x = torch.tensor(input_np)
input_y = torch.tensor(input_np).type(torch.int64)
output = input_x.to(input_y)
print(output.dtype)
# torch.int64

# MindSpore目前无法支持该功能。
```
