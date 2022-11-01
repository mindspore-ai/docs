# 比较与torch.argmin的功能差异

## torch.argmin

```text
torch.argmin(input, dim=None, keepdim=False) -> Tensor
```

更多内容详见 [torch.argmin](https://pytorch.org/docs/1.8.1/generated/torch.argmin.html)。

## mindspore.ops.Argmin

```text
class mindspore.ops.Argmin(axis=-1, output_type=mstype.int32)(input_x) -> Tensor
```

更多内容详见 [mindspore.ops.Argmin](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Argmin.html)。

## 差异对比

PyTorch：返回Tensor展平或沿着给定的维度最小值的索引，返回值类型为torch.int64。如果有多个最小值，则返回第一个最小值的索引。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，不过不支持保留进行计算的维度，且返回值类型为int32.

为保证二者输出类型是一致的，需使用[mindspore.ops.Cast](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Cast.html)算子将MindSpore的计算结果转换成mindspore.int64，以下每个示例均会有此步类型转换。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 | input | input_x | 都是输入Tensor |
| 参数 | 参数1 | dim | axis | 功能一致，参数名不同，默认值也不同 |
|  | 参数2 | keepdim | _ | PyTorch有，MindSpore无此参数，行为与PyTorch算子参数keepdim设为False时一致 |

### 代码示例1

> 对于0维的Tensor，PyTorch支持dim参数为None/-1/0及keepdim参数为True/False的任意组合，且计算结果都是一致的，都是一个0维Tensor。MindSpore 1.8.1版本暂时不支持处理0维Tensor，需要先使用[mindspore.ops.ExpandDims](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ExpandDims.html)将Tensor扩充为1维，然后再按照mindspore.ops.Argmin算子的默认参数计算。

```python
# PyTorch
import torch
import numpy as np

x = np.arange(1).reshape(()).astype(np.float32)
torch_argmin = torch.argmin
torch_output = torch_argmin(torch.tensor(x))
torch_out_np = torch_output.numpy()
print(torch_out_np)
# 0

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

x = np.arange(1).reshape(()).astype(np.float32)
ms_argmin = mindspore.ops.Argmin()
ms_expanddims = mindspore.ops.ExpandDims()
ms_cast = mindspore.ops.Cast()
ms_tensor = Tensor(x)

if not ms_tensor.shape:
    ms_tensor_tmp = ms_expanddims(ms_tensor, 0)
    ms_output = ms_argmin(ms_tensor_tmp)

ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# 0
```

### 代码示例2

> PyTorch的argmin算子在不显式给出dim参数时，计算结果是将原数组flatten后，作为1维张量做argmin操作的结果，而MindSpore仅支持对单个维度进行计算。因此，为了得到相同的计算结果，在计算前，将mindspore.ops.Argmin算子传入flatten的Tensor即可。

```python
# PyTorch
import torch
import numpy as np

x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
torch_argmin = torch.argmin
torch_output = torch_argmin(torch.tensor(x))
torch_out_np = torch_output.numpy()
print(torch_out_np)
# 0

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

dim = None
x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
ms_argmin = mindspore.ops.Argmin(
    axis=dim) if dim is not None else mindspore.ops.Argmin()
ms_expanddims = mindspore.ops.ExpandDims()
ms_cast = mindspore.ops.Cast()
ms_tensor = Tensor(x)

ms_output = ms_argmin(ms_tensor) if dim is not None else ms_argmin(
    ms_tensor.flatten())

ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# 0
```

### 代码示例3

> PyTorch算子有一个keepdim参数，当设置为True时，作用为：将进行聚合的维度保留，并设定为1。MindSpore无此参数，默认行为和PyTorch算子传入keepdim为False时一致。为了实现相同的结果，在计算完成后，使用[mindspore.ops.ExpandDims](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ExpandDims.html)算子扩充维度即可。

```python
# PyTorch
import torch
import numpy as np

dim = 1
keepdims = True
x = np.arange(2*4).reshape(2, 4).astype(np.float32)
torch_argmin = torch.argmin
torch_output = torch_argmin(torch.tensor(x), dim=dim, keepdims=keepdims)
torch_out_np = torch_output.numpy()
print(torch_out_np)
# [[0]
#  [0]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

dim = 1
keepdims = True
x = np.arange(2*4).reshape(2, 4).astype(np.float32)
ms_argmin = mindspore.ops.Argmin(axis=dim)
ms_expanddims = mindspore.ops.ExpandDims()
ms_cast = mindspore.ops.Cast()
ms_tensor = Tensor(x)

ms_output = ms_argmin(ms_tensor)
if keepdims == True:
    ms_output = ms_expanddims(ms_output, dim)

ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[0]
#  [0]]
```

### 代码示例4

> 将示例1/2/3中使用的改动都集合在一起，就能够处理所有的不对标问题了。最后，不要忘记进行类型转换。

```python
# PyTorch
import torch
import numpy as np

dim = 2
keepdims = True
x = np.arange(2*3*4).reshape(2, 3, 4).astype(np.float32)
torch_argmin = torch.argmin
torch_output = torch_argmin(torch.tensor(x), dim=dim, keepdims=keepdims)
torch_out_np = torch_output.numpy()
print(torch_out_np)
# [[[0]
#   [0]
#   [0]]
#
#  [[0]
#   [0]
#   [0]]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

dim = 2
keepdims = True
x = np.arange(2*3*4).reshape(2,3,4).astype(np.float32)
ms_argmin = mindspore.ops.Argmin(
    axis=dim) if dim is not None else mindspore.ops.Argmin()
ms_expanddims = mindspore.ops.ExpandDims()
ms_cast = mindspore.ops.Cast()
ms_tensor = Tensor(x)

if not ms_tensor.shape:
    ms_tensor_tmp = ms_expanddims(ms_tensor,0)
    ms_output = ms_argmin(ms_tensor_tmp)
else:
    ms_output = ms_argmin(ms_tensor) if dim is not None else ms_argmin(
        ms_tensor.flatten())
    if keepdims == True:
        ms_output = ms_expanddims(ms_output, dim) if dim is not None else ms_output

ms_output = ms_cast(ms_output, mindspore.int64)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[[0]
#   [0]
#   [0]]
#
#  [[0]
#   [0]
#   [0]]]
```
